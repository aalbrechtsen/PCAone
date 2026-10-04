// iobench: replay one or more winSVD passes over a PLINK .bed with different
// read strategies, without running the PCA. Measures I/O only (plus a cheap
// checksum so nothing is optimised away; --decode adds the 2-bit -> double cost).
//
// Logical order is the chunk-strided permutation: SNPs cut into runs of c,
// band b (of W) gets runs b, b+W, b+2W, ...; c=1 is permute_plink's order,
// c >= M/W is the original order. A pass reads the logical order in blocks of
// --block SNPs (like read_block_initial), merging adjacent SNPs into ranges.
//
// methods
//   seq        : one sequential read per block in file order (= reading a
//                pre-permuted file, or no permutation). Ignores W/c.
//   pread      : one pread() per contiguous range, single thread
//   pread_mt   : ranges of a block spread over --threads threads (OpenMP)
//   mmap       : memcpy ranges out of a read-only mapping
//   mmap_wn    : mmap + madvise(WILLNEED) on the next block's ranges
//   pread_pf   : pread_mt, with a background thread prefetching block i+1
//                into a second buffer while block i is "used" (double buffering)
//   window     : read the packed bytes of --kb bands at once (k * M/W SNPs of the
//                logical order). Their runs are adjacent in the file, so reads
//                are k times longer and there are k times fewer seeks. Ranges are
//                merged in file order, read with --threads preads into a staging
//                buffer, then blocks are gathered from it. RAM = k/W of the .bed.
//
// usage: iobench bed_prefix method [--W 64] [--c 1] [--block 8000] [--threads 8]
//                [--passes 1] [--random] [--fadv normal|random|seq] [--decode]
//                [--kb k] [--shuffle-ranges] [--streams] [--maxblocks n]  (time only the first n blocks of each pass, MB/s still valid)
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

using u64 = uint64_t;
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static u64 count_lines(const std::string& f) {
  std::ifstream in(f);
  u64 n = 0;
  for (std::string l; std::getline(in, l);) ++n;
  return n;
}
struct Range { u64 snp, len, dst; };  // original first snp, #snps, snp offset in block buffer

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: iobench bed_prefix method [--W 64] [--c 1] [--block 8000] [--threads 8] "
                    "[--passes 1] [--random] [--fadv normal|random|seq] [--decode]\n");
    return 1;
  }
  std::string pre = argv[1], method = argv[2], fadv = "normal";
  u64 W = 64, c = 1, block = 8000, passes = 1, maxblocks = 0, kb = 1;
  int threads = 8;
  bool rnd = false, decode = false, shufr = false, streams = false;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--W") W = std::stoull(argv[++i]);
    else if (a == "--c") c = std::stoull(argv[++i]);
    else if (a == "--block") block = std::stoull(argv[++i]);
    else if (a == "--threads") threads = std::stoi(argv[++i]);
    else if (a == "--passes") passes = std::stoull(argv[++i]);
    else if (a == "--kb") kb = std::stoull(argv[++i]);
    else if (a == "--maxblocks") maxblocks = std::stoull(argv[++i]);
    else if (a == "--random") rnd = true;
    else if (a == "--shuffle-ranges") shufr = true;
    else if (a == "--streams") streams = true;
    else if (a == "--decode") decode = true;
    else if (a == "--fadv") fadv = argv[++i];
    else { fprintf(stderr, "unknown %s\n", a.c_str()); return 1; }
  }
  const u64 M = count_lines(pre + ".bim"), N = count_lines(pre + ".fam"), bps = (N + 3) / 4;
  const std::string bed = pre + ".bed";
  omp_set_num_threads(threads);

  // logical order -> original index
  std::vector<u64> order(M);
  if (method == "seq") {
    std::iota(order.begin(), order.end(), 0);
  } else {
    u64 nruns = (M + c - 1) / c;
    std::vector<u64> runs(nruns);
    std::iota(runs.begin(), runs.end(), 0);
    if (rnd) { std::mt19937_64 g(1); std::shuffle(runs.begin(), runs.end(), g); }
    u64 k = 0;
    for (u64 b = 0; b < W; ++b)
      for (u64 r = b; r < nruns; r += W)
        for (u64 j = runs[r] * c; j < std::min(M, (runs[r] + 1) * c); ++j) order[k++] = j;
  }
  // per block: contiguous ranges
  const u64 nblocks = (M + block - 1) / block;
  std::vector<std::vector<Range>> ranges(nblocks);
  u64 nranges = 0;
  for (u64 bl = 0; bl < nblocks; ++bl) {
    u64 s = bl * block, e = std::min(M, s + block);
    for (u64 i = s; i < e; ++i) {
      auto& R = ranges[bl];
      if (!R.empty() && R.back().snp + R.back().len == order[i]) R.back().len++;
      else R.push_back({order[i], 1, i - s});
    }
    nranges += ranges[bl].size();
  }

  int fd = open(bed.c_str(), O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  if (fadv == "random") posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
  if (fadv == "seq") posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  struct stat st;
  fstat(fd, &st);
  const uint8_t* map = nullptr;
  if (method.rfind("mmap", 0) == 0) {
    map = (const uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    if (fadv == "random") madvise((void*)map, st.st_size, MADV_RANDOM);
    if (fadv == "seq") madvise((void*)map, st.st_size, MADV_SEQUENTIAL);
  }

  std::vector<uint8_t> buf(block * bps), buf2(block * bps);
  std::vector<double> G;
  if (decode) G.resize(N * block);
  volatile u64 sink = 0;

  auto read_block = [&](u64 bl, uint8_t* dst, bool par) {
    auto& R = ranges[bl];
    auto one = [&](const Range& r) {
      u64 off = 3 + r.snp * bps, len = r.len * bps;
      uint8_t* d = dst + r.dst * bps;
      if (map) { memcpy(d, map + off, len); return; }
      u64 done = 0;
      while (done < len) {
        ssize_t got = pread(fd, d + done, len - done, off + done);
        if (got <= 0) { perror("pread"); exit(1); }
        done += got;
      }
    };
    if (par) {
#pragma omp parallel for schedule(dynamic, 1)
      for (size_t i = 0; i < R.size(); ++i) one(R[i]);
    } else {
      for (auto& r : R) one(r);
    }
  };
  auto use_block = [&](u64 bl, const uint8_t* src) {
    u64 nb = std::min(M, (bl + 1) * block) - bl * block;
    if (!decode) {
      u64 s = 0;
      for (u64 i = 0; i < nb * bps; i += 4096) s += src[i];
      sink = sink + s;
      return;
    }
    static const double LUT[4] = {2, 9, 1, 0};
#pragma omp parallel for
    for (u64 i = 0; i < nb; ++i)
      for (u64 j = 0; j < N; ++j) G[i * N + j] = LUT[(src[i * bps + j / 4] >> ((j & 3) * 2)) & 3];
    sink = sink + (u64)G[0];
  };

  const u64 runblocks = maxblocks ? std::min(maxblocks, nblocks) : nblocks;
  u64 bytes_read = 0, nwin_ranges = 0;
  for (u64 bl = 0; bl < runblocks; ++bl) bytes_read += (std::min(M, (bl + 1) * block) - bl * block) * bps;

  // persistent prefetch thread for pread_pf (keeps its own OpenMP pool)
  std::mutex mu;
  std::condition_variable cv;
  long want = -1, have = -1, want_bl = 0;
  bool quit = false;
  std::thread pf;
  if (method == "pread_pf")
    pf = std::thread([&] {
      omp_set_num_threads(threads);
      for (;;) {
        long bl;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return quit || want > have; });
          if (quit) return;
          bl = want;
        }
        read_block((u64)want_bl, buf2.data(), true);
        {
          std::lock_guard<std::mutex> lk(mu);
          have = bl;
        }
        cv.notify_all();
      }
    });

  double t0 = now();
  for (u64 p = 0; p < passes; ++p) {
    if (method == "seq") {
      for (u64 bl = 0; bl < runblocks; ++bl) {
        u64 nb = std::min(M, (bl + 1) * block) - bl * block;
        u64 off = 3 + bl * block * bps, len = nb * bps, done = 0;
        while (done < len) {
          ssize_t got = pread(fd, buf.data() + done, len - done, off + done);
          if (got <= 0) { perror("pread"); return 1; }
          done += got;
        }
        use_block(bl, buf.data());
      }
    } else if (method == "window") {
      // a window = kb bands = kb * ceil(M/W) logical SNPs; snps of a window, in file order
      const u64 wsnps = kb * ((M + W - 1) / W);
      const u64 runsnps = runblocks * block < M ? runblocks * block : M;
      std::vector<uint8_t> stage;
      for (u64 w0 = 0; w0 < runsnps; w0 += wsnps) {
        u64 w1 = std::min(runsnps, w0 + wsnps);
        std::vector<std::pair<u64, u64>> idx;  // (original snp, logical position)
        idx.reserve(w1 - w0);
        for (u64 i = w0; i < w1; ++i) idx.push_back({order[i], i - w0});
        std::sort(idx.begin(), idx.end());
        std::vector<Range> R;  // contiguous file ranges; dst = slot in stage (file order)
        for (u64 i = 0; i < idx.size(); ++i) {
          if (!R.empty() && R.back().snp + R.back().len == idx[i].first) R.back().len++;
          else R.push_back({idx[i].first, 1, i});
        }
        stage.resize((w1 - w0) * bps);
        nwin_ranges += R.size();
        if (shufr) { std::mt19937_64 g(w0); std::shuffle(R.begin(), R.end(), g); }  // spread concurrent reads over the disk
        auto rd = [&](size_t i) {
          u64 off = 3 + R[i].snp * bps, len = R[i].len * bps, done = 0;
          uint8_t* d = stage.data() + R[i].dst * bps;
          while (done < len) {
            ssize_t got = pread(fd, d + done, len - done, off + done);
            if (got <= 0) { perror("pread"); exit(1); }
            done += got;
          }
        };
        if (streams) {  // each thread reads one contiguous slice of the sorted ranges, in order
#pragma omp parallel for schedule(static)
          for (size_t i = 0; i < R.size(); ++i) rd(i);
        } else {
#pragma omp parallel for schedule(dynamic, 1)
          for (size_t i = 0; i < R.size(); ++i) rd(i);
        }
        // gather logical blocks from the stage and use them
        std::vector<u64> slot(w1 - w0);
        for (u64 i = 0; i < idx.size(); ++i) slot[idx[i].second] = i;
        for (u64 s0 = w0; s0 < w1; s0 += block) {
          u64 s1 = std::min(w1, s0 + block);
#pragma omp parallel for
          for (u64 i = s0; i < s1; ++i) memcpy(buf.data() + (i - s0) * bps, stage.data() + slot[i - w0] * bps, bps);
          use_block(s0 / block, buf.data());
        }
      }
    } else if (method == "pread_pf") {
      read_block(0, buf.data(), true);
      for (u64 bl = 0; bl < runblocks; ++bl) {
        if (bl + 1 < runblocks) {
          std::lock_guard<std::mutex> lk(mu);
          want_bl = (long)(bl + 1);
          want = (long)(p * nblocks + bl + 1);  // unique request id across passes
        }
        cv.notify_all();
        use_block(bl, buf.data());
        if (bl + 1 < runblocks) {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return have == want; });
          std::swap(buf, buf2);
        }
      }
    } else {
      for (u64 bl = 0; bl < runblocks; ++bl) {
        if (method == "mmap_wn" && bl + 1 < runblocks)
          for (auto& r : ranges[bl + 1]) {
            u64 off = 3 + r.snp * bps, a = off & ~4095ULL;
            madvise((void*)(map + a), off + r.len * bps - a, MADV_WILLNEED);
          }
        read_block(bl, buf.data(), method == "pread_mt" || method == "mmap" || method == "mmap_wn");
        use_block(bl, buf.data());
      }
    }
  }
  double t = now() - t0;
  if (pf.joinable()) {
    { std::lock_guard<std::mutex> lk(mu); quit = true; }
    cv.notify_all();
    pf.join();
  }
  double gb = (double)bytes_read * passes / 1e9;
  if (method == "window" && shufr) method += "S";
  if (method == "window" && streams) method += "T";
  if (method.rfind("window", 0) == 0) method += "_k" + std::to_string(kb) + "(ranges/win=" + std::to_string(nwin_ranges / std::max<u64>(1, (passes * ((std::min(M, runblocks * block) + kb * ((M + W - 1) / W) - 1) / (kb * ((M + W - 1) / W)))))) + ")";
  printf("%s\tW=%llu\tc=%llu%s\tblock=%llu\tthr=%d\tfadv=%s\tpasses=%llu\tranges/block=%.1f\trun_KB=%.1f\tblocks=%llu/%llu\t%.2f s\t%.0f MB/s\n",
         method.c_str(), (unsigned long long)W, (unsigned long long)c, rnd ? "r" : "", (unsigned long long)block,
         threads, fadv.c_str(), (unsigned long long)passes, (double)nranges / nblocks, c * bps / 1024.0, (unsigned long long)runblocks, (unsigned long long)nblocks, t,
         gb * 1000 / t);
  return 0;
}
