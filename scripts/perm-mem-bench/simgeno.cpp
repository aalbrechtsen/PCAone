// simgeno: simulate PLINK bed/bim/fam with population structure and local LD.
//
// Model (haplotype-block caricature of the coalescent)
//  * --nhap ancestral haplotypes; allele at each SNP ~ Bernoulli(p), p ~ U(0.05, 0.95).
//  * The genome is cut into LD blocks of geometric length, mean --blocklen SNPs
//    (shared by everyone, like recombination hotspots).
//  * In each block, population k has its own frequencies w_k over the ancestral
//    haplotypes: w_k ~ Dirichlet(c * base), base ~ Dirichlet(1), c = (1-F)/F with
//    F = --fst. This gives population differentiation.
//  * A sample haplotype picks, per block, an ancestral haplotype from w of its
//    current ancestry. Strong LD within blocks, none across block boundaries.
//  * Ancestry along a haplotype switches with mean tract length --tractlen SNPs,
//    drawn from the individual's admixture proportions q. A fraction --admixed
//    of individuals have q ~ Dirichlet(1,..,1), the rest are unadmixed.
//  * --inv regions of --invlen SNPs mimic inversions: two haplotype classes with
//    different alleles at half the SNPs, class frequency differing by population.
//    This creates strong, long-range local LD - the case where contiguous winSVD
//    mini-batches go wrong.
//  * Allele copy error --eps, missingness --miss.
//
// SNPs are split evenly across --chr chromosomes; LD and ancestry reset at each.
// FID in the .fam is the population of the largest ancestry component.
//
// Build: g++ -O3 -march=native -fopenmp -o simgeno simgeno.cpp

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct Rng {  // xoshiro256**
  uint64_t s[4];
  explicit Rng(uint64_t seed) {
    for (int i = 0; i < 4; ++i) {  // splitmix64
      seed += 0x9e3779b97f4a7c15ULL;
      uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      s[i] = z ^ (z >> 31);
    }
  }
  static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  uint64_t next() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0], s[3] ^= s[1], s[1] ^= s[2], s[0] ^= s[3], s[2] ^= t, s[3] = rotl(s[3], 45);
    return r;
  }
  double unif() { return (next() >> 11) * 0x1.0p-53; }
  uint32_t below(uint32_t n) { return (uint32_t)((next() >> 32) * n >> 32); }
  // geometric distance (>=1) to the next event with per-step probability prob
  uint64_t geom(double prob) {
    if (prob <= 0) return UINT64_MAX;
    double u = unif();
    return 1 + (uint64_t)(std::log1p(-u) / std::log1p(-prob));
  }
};

struct Opts {
  uint64_t n = 10000, m = 1000000;
  int chr = 22, K = 3, nhap = 16, inv = 3;
  double fst = 0.05, blocklen = 100, tractlen = 5000, admixed = 0.2, eps = 0.01, miss = 0.002;
  uint64_t invlen = 3000;
  uint64_t seed = 1;
  std::string out = "sim";
};

static void usage() {
  fprintf(stderr,
          "simgeno -o prefix [-n samples] [-m snps] [--chr 22] [-K 3] [--nhap 16] [--fst 0.05]\n"
          "        [--blocklen 100] [--tractlen 5000] [--admixed 0.2] [--inv 3] [--invlen 3000]\n"
          "        [--eps 0.01] [--miss 0.002] [--seed 1]\n");
  exit(1);
}

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (i + 1 >= argc) usage();
    const char* v = argv[++i];
    if (a == "-o") o.out = v;
    else if (a == "-n") o.n = std::stoull(v);
    else if (a == "-m") o.m = std::stoull(v);
    else if (a == "--chr") o.chr = std::stoi(v);
    else if (a == "-K") o.K = std::stoi(v);
    else if (a == "--nhap") o.nhap = std::stoi(v);
    else if (a == "--fst") o.fst = std::stod(v);
    else if (a == "--blocklen") o.blocklen = std::stod(v);
    else if (a == "--tractlen") o.tractlen = std::stod(v);
    else if (a == "--admixed") o.admixed = std::stod(v);
    else if (a == "--inv") o.inv = std::stoi(v);
    else if (a == "--invlen") o.invlen = std::stoull(v);
    else if (a == "--eps") o.eps = std::stod(v);
    else if (a == "--miss") o.miss = std::stod(v);
    else if (a == "--seed") o.seed = std::stoull(v);
    else usage();
  }
  const uint64_t n = o.n, m = o.m;
  const int K = o.K, A = o.nhap;
  const uint64_t bps = (n + 3) / 4;  // bed bytes per snp
  fprintf(stderr, "simgeno: n=%llu m=%llu K=%d nhap=%d fst=%g blocklen=%g bed=%.2f GB\n",
          (unsigned long long)n, (unsigned long long)m, K, A, o.fst, o.blocklen, (double)m * bps / 1e9);

  // chromosome of each snp
  std::vector<uint64_t> chr_start(o.chr + 1);
  for (int c = 0; c <= o.chr; ++c) chr_start[c] = m * c / o.chr;

  // ancestral haplotypes: hap[j*A + a]
  std::vector<uint8_t> hap(m * A);
#pragma omp parallel
  {
    Rng r(o.seed * 1000003 + omp_get_thread_num());
#pragma omp for schedule(static)
    for (uint64_t j = 0; j < m; ++j) {
      double p = 0.05 + 0.9 * r.unif();
      for (int a = 0; a < A; ++a) hap[j * A + a] = r.unif() < p;
    }
  }

  // LD blocks (never crossing a chromosome boundary) and per-block, per-population
  // cumulative haplotype weights cw[(blk*K + k)*A + a]
  std::vector<uint32_t> block_of(m);
  std::vector<double> cw;
  {
    std::mt19937_64 g(o.seed * 31 + 5);
    std::gamma_distribution<double> G1(1, 1);
    const double conc = (1 - o.fst) / o.fst;
    Rng r(o.seed * 77 + 1);
    uint32_t blk = 0;
    std::vector<double> base(A), w(A);
    for (int c = 0; c < o.chr; ++c) {
      uint64_t j = chr_start[c];
      while (j < chr_start[c + 1]) {
        uint64_t len = r.geom(1.0 / o.blocklen);
        uint64_t e = std::min(chr_start[c + 1], j + len);
        for (uint64_t t = j; t < e; ++t) block_of[t] = blk;
        double s = 0;
        for (int a = 0; a < A; ++a) s += (base[a] = G1(g));
        for (int a = 0; a < A; ++a) base[a] /= s;
        for (int k = 0; k < K; ++k) {
          s = 0;
          for (int a = 0; a < A; ++a)
            s += (w[a] = std::gamma_distribution<double>(std::max(1e-3, conc * base[a]), 1)(g));
          double cum = 0;
          for (int a = 0; a < A; ++a) cw.push_back(cum += w[a] / s);
          cw.back() = 1.0;
        }
        ++blk;
        j = e;
      }
    }
    fprintf(stderr, "simgeno: %u LD blocks\n", blk);
  }

  // inversion-like regions: class consensus haplotypes and class-1 freq per pop
  struct Inv { uint64_t start, len; std::vector<uint8_t> cons[2]; std::vector<double> f1; };
  std::vector<Inv> invs;
  std::vector<int> inv_of(m, -1);
  {
    Rng r(o.seed ^ 0xabcdef);
    for (int i = 0; i < o.inv; ++i) {
      int c = (int)((uint64_t)(i + 1) * o.chr / (o.inv + 1));  // spread over chromosomes
      uint64_t cs = chr_start[c], ce = chr_start[c + 1];
      uint64_t len = std::min<uint64_t>(o.invlen, (ce - cs) / 2);
      Inv v;
      v.start = cs + (ce - cs - len) / 2;
      v.len = len;
      for (int t = 0; t < 2; ++t) v.cons[t].resize(len);
      for (uint64_t j = 0; j < len; ++j) {
        uint8_t a = r.unif() < 0.5;
        v.cons[0][j] = a;
        v.cons[1][j] = (r.unif() < 0.5) ? (uint8_t)(1 - a) : a;
        inv_of[v.start + j] = (int)invs.size();
      }
      for (int k = 0; k < K; ++k) v.f1.push_back(0.1 + 0.8 * r.unif());
      invs.push_back(std::move(v));
    }
  }

  // admixture proportions
  std::vector<double> Q(n * K);
  std::vector<int> label(n);
  {
    std::mt19937_64 g(o.seed * 7 + 3);
    std::uniform_real_distribution<double> U01(0, 1);
    std::gamma_distribution<double> G1(1, 1);
    for (uint64_t i = 0; i < n; ++i) {
      double* q = &Q[i * K];
      if (U01(g) < o.admixed) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += (q[k] = G1(g));
        for (int k = 0; k < K; ++k) q[k] /= s;
      } else {
        for (int k = 0; k < K; ++k) q[k] = 0;
        q[i % K] = 1;
      }
      label[i] = (int)(std::max_element(q, q + K) - q);
    }
  }

  std::vector<uint8_t> bed(m * bps, 0);
  const double p_tract = 1.0 / o.tractlen;
  const uint64_t G = 64;  // samples per task (16 bytes per snp)
  const uint64_t ngroups = (n + G - 1) / G;

#pragma omp parallel for schedule(dynamic, 1)
  for (uint64_t grp = 0; grp < ngroups; ++grp) {
    Rng r(o.seed * 0x1234567ULL + grp * 0x9e37ULL + 17);
    uint64_t s0 = grp * G, s1 = std::min(n, s0 + G), ns = s1 - s0;
    // per haplotype state: ancestry, ancestral haplotype copied, inversion class
    std::vector<int> anc(2 * ns), ah(2 * ns), cls(2 * ns);
    std::vector<uint64_t> next_tract(2 * ns);
    auto draw_anc = [&](uint64_t i) {
      double u = r.unif(), c = 0;
      const double* q = &Q[i * K];
      for (int k = 0; k < K; ++k)
        if ((c += q[k]) > u) return k;
      return K - 1;
    };
    auto draw_hap = [&](uint32_t blk, int k) {
      const double* c = &cw[((uint64_t)blk * K + k) * A];
      double u = r.unif();
      return (int)(std::upper_bound(c, c + A, u) - c);
    };
    for (int c = 0; c < o.chr; ++c) {
      uint64_t cs = chr_start[c], ce = chr_start[c + 1];
      for (uint64_t h = 0; h < 2 * ns; ++h) {
        anc[h] = draw_anc(s0 + h / 2);
        next_tract[h] = cs + r.geom(p_tract);
        cls[h] = 0;
      }
      for (uint64_t j = cs; j < ce; ++j) {
        int iv = inv_of[j];
        bool inv_start = iv >= 0 && invs[iv].start == j;
        bool new_block = j == cs || block_of[j] != block_of[j - 1];
        uint8_t* row = &bed[j * bps];
        for (uint64_t s = 0; s < ns; ++s) {
          int g = 0;
          for (int t = 0; t < 2; ++t) {
            uint64_t h = 2 * s + t;
            bool redraw = new_block;
            if (j >= next_tract[h]) {
              anc[h] = draw_anc(s0 + s);
              next_tract[h] = j + r.geom(p_tract);
              redraw = true;
            }
            if (redraw) ah[h] = std::min(A - 1, draw_hap(block_of[j], anc[h]));
            if (inv_start) cls[h] = r.unif() < invs[iv].f1[anc[h]];
            uint8_t a = hap[j * A + ah[h]];
            if (iv >= 0 && r.unif() < 0.95) a = invs[iv].cons[cls[h]][j - invs[iv].start];
            if (r.unif() < o.eps) a ^= 1;
            g += a;
          }
          // bed: 00 hom A1, 10 het, 11 hom A2, 01 missing. g = copies of A1.
          uint8_t code = g == 2 ? 0 : (g == 1 ? 2 : 3);
          if (o.miss > 0 && r.unif() < o.miss) code = 1;
          uint64_t i = s0 + s;
          row[i >> 2] |= code << ((i & 3) * 2);
        }
      }
    }
  }

  // write files
  {
    std::ofstream f(o.out + ".bed", std::ios::binary);
    const uint8_t magic[3] = {0x6c, 0x1b, 0x01};
    f.write((const char*)magic, 3);
    f.write((const char*)bed.data(), bed.size());
  }
  {
    FILE* f = fopen((o.out + ".bim").c_str(), "w");
    for (int c = 0; c < o.chr; ++c)
      for (uint64_t j = chr_start[c]; j < chr_start[c + 1]; ++j)
        fprintf(f, "%d\tsnp%llu\t0\t%llu\tA\tG\n", c + 1, (unsigned long long)j,
                (unsigned long long)(1000 + (j - chr_start[c]) * 100));
    fclose(f);
  }
  {
    FILE* f = fopen((o.out + ".fam").c_str(), "w");
    for (uint64_t i = 0; i < n; ++i) fprintf(f, "pop%d\tind%llu\t0\t0\t0\t-9\n", label[i] + 1, (unsigned long long)i);
    fclose(f);
  }
  {
    FILE* f = fopen((o.out + ".Q").c_str(), "w");
    for (uint64_t i = 0; i < n; ++i) {
      for (int k = 0; k < K; ++k) fprintf(f, "%s%.5f", k ? " " : "", Q[i * K + k]);
      fprintf(f, "\n");
    }
    fclose(f);
  }
  {
    FILE* f = fopen((o.out + ".inv").c_str(), "w");
    fprintf(f, "start\tlen\n");
    for (auto& v : invs) fprintf(f, "%llu\t%llu\n", (unsigned long long)v.start, (unsigned long long)v.len);
    fclose(f);
  }
  fprintf(stderr, "simgeno: wrote %s.{bed,bim,fam,Q,inv}\n", o.out.c_str());
  return 0;
}
