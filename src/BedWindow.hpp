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
  BedWindowReader(const std::string& bed, uint64_t width, std::vector<uint32_t> order, std::vector<uint64_t> starts,
                  int io_threads)
      : width_(width), order_(std::move(order)), starts_(std::move(starts)), nio_(std::max(1, io_threads)) {
    if (starts_.size() < 2 || starts_.front() != 0 || starts_.back() != order_.size())
      throw std::invalid_argument("BedWindowReader: windows must cover the logical SNPs");
    fd_ = ::open(bed.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Cannot open " + bed);
  }

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
    if (cur_.id != w) {
      auto t0 = std::chrono::steady_clock::now();
      if (pending_.valid() && pending_id_ == w) {
        pending_.get();
        std::swap(cur_, next_);
      } else {
        if (pending_.valid()) pending_.get();
        load(w, cur_);
      }
      pending_id_ = -1;
      wait_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
  std::vector<uint64_t> starts_;
  int nio_;
  Window cur_, next_;
  std::future<void> pending_;
  long pending_id_ = -1;
  std::atomic<bool> stop_{false};
  uint64_t bytes_read_ = 0;  // updated by the loading thread; read it after finish()
  double wait_seconds_ = 0;
};

}  // namespace PCAone
#endif
