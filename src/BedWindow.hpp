#ifndef PCAONE_BED_WINDOW_HPP
#define PCAONE_BED_WINDOW_HPP

#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace PCAone {

// Reads a PLINK BED in a logical (permuted) SNP order without rewriting it.
//
// order[d] is the source SNP of logical SNP d. The logical SNPs are cut into
// windows [starts[w], starts[w + 1]). A window is read in one go: its source
// SNPs are sorted, adjacent ones merged into contiguous ranges, and the ranges
// read with many concurrent pread() calls in file order (a spinning RAID serves
// those far better than one stream of short reads). While one window is used,
// the next is read in the background. With a single window the whole BED is
// read once and kept, so later passes need no I/O at all.
//
// With the interleaved permutation (band b = SNPs b, b+W, b+2W, ...), a window
// of k consecutive bands reads contiguous pieces of k SNPs, so the window size
// trades RAM for read length.
class BedWindowReader {
 public:
  // bands[b] is the first logical SNP of band b (bands.back() = number of SNPs).
  // A window is k consecutive bands. With grow, k doubles (up to kmax) after a
  // pass in which the caller waited for I/O more than 5 % of the pass.
  BedWindowReader(const std::string& bed, uint64_t width, std::vector<uint32_t> order, std::vector<uint64_t> bands,
                  uint64_t k, uint64_t kmax, bool grow, int io_threads,
                  std::function<void(const std::string&)> log = nullptr)
      : width_(width), order_(std::move(order)), bands_(std::move(bands)), k_(std::max<uint64_t>(1, k)),
        kmax_(std::max<uint64_t>(k_, kmax)), grow_(grow), nio_(std::max(1, io_threads)), log_(std::move(log)) {
    if (bands_.size() < 2 || bands_.front() != 0 || bands_.back() != order_.size())
      throw std::invalid_argument("BedWindowReader: bands must cover the logical SNPs");
    set_windows();
    fd_ = ::open(bed.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Cannot open " + bed);
  }

  uint64_t bands_per_window() const { return k_; }
  uint64_t max_bands_per_window() const { return kmax_seen_; }

  ~BedWindowReader() {
    finish();
    if (fd_ >= 0) ::close(fd_);
  }

  // Stop and wait for a background read; e.g. the read of the first window
  // started after the last block of the last pass.
  void finish() {
    if (!pending_.valid()) return;
    stop_ = true;
    try {
      pending_.get();
    } catch (...) {
    }
    stop_ = false;
    pending_id_ = -1;
  }

  BedWindowReader(const BedWindowReader&) = delete;
  BedWindowReader& operator=(const BedWindowReader&) = delete;

  uint64_t nwindows() const { return starts_.size() - 1; }
  uint64_t bytes_read() const { return bytes_read_; }
  double wait_seconds() const { return wait_seconds_; }

  // Copy the records of logical SNPs [first, first + count) into dst
  // (count * width bytes). They must lie in one window.
  void copy(uint64_t first, uint64_t count, unsigned char* dst) {
    const long w = (long)(std::upper_bound(starts_.begin(), starts_.end(), first) - starts_.begin()) - 1;
    if (w < 0 || first + count > starts_[w + 1]) throw std::logic_error("BedWindowReader: block crosses a window");
    if (w == 0 && cur_.id != 0) {  // a new pass begins
      const auto now = std::chrono::steady_clock::now();
      if (pass_started_ && grow_ && k_ < kmax_ && nwindows() > 1) {
        const double pass = std::chrono::duration<double>(now - pass_start_).count();
        const double waited = stall_seconds_ - stall_at_pass_start_;
        if (waited > 0.05 * pass) {
          finish();
          cur_ = Window();  // release the old windows before larger ones are read
          next_ = Window();
          const uint64_t old = k_;
          k_ = std::min(kmax_, 2 * k_);
          set_windows();
          if (log_)
            log_("waited " + std::to_string(waited) + " s for I/O in a " + std::to_string(pass) +
                 " s pass; window " + std::to_string(old) + " -> " + std::to_string(k_) + " bands");
        }
      }
      pass_started_ = true;
      pass_start_ = now;
      stall_at_pass_start_ = stall_seconds_;
    }
    if (cur_.id != w) {
      auto t0 = std::chrono::steady_clock::now();
      // a stall: waiting for a background read, or reading in the foreground
      // although a previous window was in use (so it could have been read
      // ahead). The first read, and the first after the windows changed, are
      // unavoidable and do not count towards --perm-adapt.
      const bool avoidable = cur_.id >= 0;
      if (pending_.valid() && pending_id_ == w) {
        pending_.get();
        std::swap(cur_, next_);
      } else {
        if (pending_.valid()) pending_.get();
        load(w, cur_);
      }
      pending_id_ = -1;
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      wait_seconds_ += dt;
      if (avoidable) stall_seconds_ += dt;
    }
    // read the next window (the first again after the last, for the next pass)
    const long nw = (long)nwindows();
    if (nw > 1 && !pending_.valid()) {
      pending_id_ = (w + 1) % nw;
      pending_ = std::async(std::launch::async, [this] { load(pending_id_, next_); });
    }
    const uint64_t base = starts_[w];
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)count; ++i)
      std::memcpy(dst + i * width_, cur_.data.data() + (uint64_t)cur_.slot[first - base + i] * width_, width_);
  }

 private:
  void set_windows() {
    starts_.clear();
    const uint64_t nb = bands_.size() - 1;
    for (uint64_t b = 0; b < nb; b += k_)
      if (bands_[b] < bands_.back() && (starts_.empty() || bands_[b] > starts_.back())) starts_.push_back(bands_[b]);
    starts_.push_back(bands_.back());
    kmax_seen_ = std::max(kmax_seen_, k_);
  }

  struct Window {
    long id = -1;
    std::vector<unsigned char> data;  // records in file order
    std::vector<uint32_t> slot;       // logical SNP in window -> record in data
  };

  void load(long w, Window& win) {
    const uint64_t first = starts_[w], n = starts_[w + 1] - first;
    std::vector<std::pair<uint32_t, uint32_t>> src(n);  // (source SNP, logical SNP in window)
    for (uint64_t i = 0; i < n; ++i) src[i] = {order_[first + i], (uint32_t)i};
    std::sort(src.begin(), src.end());
    struct Range {
      uint64_t snp, len, pos;
    };
    std::vector<Range> ranges;
    win.slot.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
      win.slot[src[i].second] = (uint32_t)i;
      if (!ranges.empty() && ranges.back().snp + ranges.back().len == src[i].first)
        ++ranges.back().len;
      else
        ranges.push_back({src[i].first, 1, i});
    }
    win.data.resize(n * width_);
    bool failed = false;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nio_)
    for (int64_t r = 0; r < (int64_t)ranges.size(); ++r) {
      if (stop_) continue;
      const uint64_t off = 3 + ranges[r].snp * width_, len = ranges[r].len * width_;
      unsigned char* d = win.data.data() + ranges[r].pos * width_;
      uint64_t done = 0;
      while (done < len) {
        const ssize_t got = ::pread(fd_, d + done, len - done, off + done);
        if (got <= 0) {
#pragma omp atomic write
          failed = true;
          break;
        }
        done += got;
      }
    }
    if (failed) throw std::runtime_error("Short read from the BED file");
    if (stop_) {
      win.id = -1;
      return;
    }
    win.id = w;
    bytes_read_ += n * width_;
  }

  int fd_ = -1;
  uint64_t width_;
  std::vector<uint32_t> order_;
  std::vector<uint64_t> bands_, starts_;
  uint64_t k_, kmax_, kmax_seen_ = 0;
  bool grow_;
  int nio_;
  std::function<void(const std::string&)> log_;
  bool pass_started_ = false;
  std::chrono::steady_clock::time_point pass_start_;
  double stall_seconds_ = 0, stall_at_pass_start_ = 0;
  Window cur_, next_;
  std::future<void> pending_;
  long pending_id_ = -1;
  std::atomic<bool> stop_{false};
  uint64_t bytes_read_ = 0;  // updated by the loading thread; read it after finish()
  double wait_seconds_ = 0;
};

}  // namespace PCAone
#endif
