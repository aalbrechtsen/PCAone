/*******************************************************************************
 * pcaone-ibd -- IBD sharing probabilities (k0, k1, k2) for related pairs.
 *
 * Reads PCAone output plus the PLINK genotypes; shares no state with PCAone.
 *
 *   PCAone     -b plink -k <K-1> --evaladmix -o pcs
 *   pcaone-ibd -b plink -P pcs --min-kin 0.05 -o rel
 *
 * Kinship alone cannot identify a relationship: parent-offspring and full sibs
 * both have phi = 1/4, and only k0 separates them (0 vs 1/4).
 *
 * Method. Individual allele frequencies are recovered from the PC scores and
 * the genotypes,
 *
 *     pi_is = 0.5 * [ V (V'V)^-1 V' G ]_is ,   V = [PC_1..PC_k, 1]
 *
 * and (k0,k1,k2) are fitted per pair by EM over the three IBD states. The
 * likelihood is written at the ALLELE level, not the genotype level:
 *
 *     IBD=0:  g_i ~ HWE(pi_i),  g_j ~ HWE(pi_j),  independent
 *     IBD=1:  shared allele S ~ pi_ij ;  i's other ~ pi_i ;  j's other ~ pi_j
 *     IBD=2:  g_i = g_j ~ HWE(pi_ij)
 *
 * with pi_ij = (pi_i + pi_j)/2. Keeping pi_i and pi_j apart is essential, not a
 * refinement: relatives in admixed data frequently differ in ancestry (half-sib
 * pairs in the PCAone benchmark differ by 0.43 in admixture proportion), and
 * pooling them to a single frequency reports a parent-offspring pair as 83%
 * unrelated.
 *
 * Only pairs above --min-kin are estimated. k is meaningless for unrelated
 * pairs, and the restriction is what makes a per-pair likelihood affordable:
 * related pairs are O(N), not O(N^2).
 ******************************************************************************/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "../external/Eigen/Dense"

using Mat2D = Eigen::MatrixXd;
using Mat1D = Eigen::VectorXd;

static void die(const std::string& m) { std::cerr << "error: " << m << "\n"; std::exit(1); }

static const double BED2GENO[4] = {2.0, -9.0, 1.0, 0.0};  // 00=hom A1, 01=missing, 10=het, 11=hom A2

struct Args {
  std::string plink, pref, out = "pcaone-ibd";
  double min_kin = 0.05;
  int npc = 0, threads = 4;
  bool loo = true;
};

static void usage() {
  std::cout <<
    "pcaone-ibd -- IBD sharing probabilities (k0,k1,k2) for related pairs\n\n"
    "usage: pcaone-ibd -b <plink prefix> -P <pcaone prefix> [options]\n\n"
    "  -b <prefix>      PLINK .bed/.bim/.fam prefix\n"
    "  -P <prefix>      PCAone output prefix (.eigvecs, .kinship, .mbim)\n"
    "  -o <prefix>      output prefix (default pcaone-ibd)\n"
    "  --min-kin <f>    only estimate pairs with kinship above this (default 0.05)\n"
    "  -k <int>         number of PCs to use (default: all in .eigvecs)\n"
    "  -n <int>         threads (default 4)\n"
    "  --no-loo         do not leave the pair out of the allele-frequency fit\n";
  std::exit(0);
}

// ------------------------------------------------------------------ readers --
static std::vector<std::string> read_col2(const std::string& path, int col, size_t& n) {
  std::ifstream f(path);
  if (!f.is_open()) die("cannot open " + path);
  std::vector<std::string> v;
  std::string line, tok;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream is(line);
    std::string got;
    for (int c = 0; c <= col; ++c)
      if (!(is >> tok)) die("too few columns in " + path);
    got = tok;
    v.push_back(got);
  }
  n = v.size();
  return v;
}

static Mat2D read_matrix(const std::string& path, bool skip_header) {
  std::ifstream f(path);
  if (!f.is_open()) die("cannot open " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  if (skip_header) std::getline(f, line);
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    std::istringstream is(line);
    std::vector<double> r;
    double v;
    while (is >> v) r.push_back(v);
    if (!r.empty()) rows.push_back(std::move(r));
  }
  if (rows.empty()) die("no numeric rows in " + path);
  Mat2D M(rows.size(), rows[0].size());
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i].size() != (size_t)M.cols()) die("ragged matrix in " + path);
    for (Eigen::Index j = 0; j < M.cols(); ++j) M(i, j) = rows[i][j];
  }
  return M;
}

// ------------------------------------------------------------- the estimator --
static inline double dg(int g, double p) {              // HWE genotype probability
  const double q = 1.0 - p;
  return g == 0 ? q * q : (g == 1 ? 2.0 * p * q : p * p);
}
static inline double aI(int g, int S, double p) {       // the non-shared allele
  const int d = g - S;
  return d == 0 ? 1.0 - p : (d == 1 ? p : 0.0);
}

// EM over the three IBD states. lik is 3 x nsite, column-major per site.
static void em_ibd(const std::vector<double>& lik, size_t ns, double* k) {
  k[0] = k[1] = k[2] = 1.0 / 3.0;
  for (int it = 0; it < 500; ++it) {
    double a0 = 0, a1 = 0, a2 = 0;
    size_t used = 0;
    for (size_t s = 0; s < ns; ++s) {
      const double w0 = k[0] * lik[3 * s], w1 = k[1] * lik[3 * s + 1], w2 = k[2] * lik[3 * s + 2];
      const double t = w0 + w1 + w2;
      if (!(t > 0)) continue;
      a0 += w0 / t; a1 += w1 / t; a2 += w2 / t; ++used;
    }
    if (used == 0) { k[0] = 1; k[1] = k[2] = 0; return; }
    const double n0 = a0 / used, n1 = a1 / used, n2 = a2 / used;
    const double d = std::fabs(n0 - k[0]) + std::fabs(n1 - k[1]) + std::fabs(n2 - k[2]);
    k[0] = n0; k[1] = n1; k[2] = n2;
    if (d < 1e-9) break;
  }
}

int main(int argc, char** argv) {
  Args a;
  if (argc == 1) usage();
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() { if (i + 1 >= argc) die("missing value for " + s); return std::string(argv[++i]); };
    if (s == "-b") a.plink = next();
    else if (s == "-P") a.pref = next();
    else if (s == "-o") a.out = next();
    else if (s == "--min-kin") a.min_kin = std::stod(next());
    else if (s == "-k") a.npc = std::stoi(next());
    else if (s == "-n") a.threads = std::stoi(next());
    else if (s == "--no-loo") a.loo = false;
    else if (s == "-h" || s == "--help") usage();
    else die("unknown option " + s);
  }
  if (a.plink.empty() || a.pref.empty()) die("-b and -P are required");
#ifdef _OPENMP
  omp_set_num_threads(a.threads);
#endif

  // ---- samples, sites, and the site filter the PCA used --------------------
  size_t N = 0, Mall = 0, Mkeep = 0;
  std::vector<std::string> iid = read_col2(a.plink + ".fam", 1, N);
  std::vector<std::string> bimid = read_col2(a.plink + ".bim", 1, Mall);
  std::vector<std::string> mbid = read_col2(a.pref + ".mbim", 1, Mkeep);
  std::cout << "samples " << N << ", sites in .bim " << Mall << ", sites used by the PCA " << Mkeep << "\n";

  std::unordered_map<std::string, size_t> want;
  want.reserve(Mkeep * 2);
  for (size_t j = 0; j < Mkeep; ++j) want.emplace(mbid[j], j);
  std::vector<int> keep_idx(Mall, -1);
  size_t matched = 0;
  for (size_t j = 0; j < Mall; ++j) {
    auto it = want.find(bimid[j]);
    if (it != want.end()) { keep_idx[j] = (int)it->second; ++matched; }
  }
  if (matched != Mkeep)
    die("only " + std::to_string(matched) + " of " + std::to_string(Mkeep) +
        " sites in .mbim were found in .bim -- is -P from the same dataset?");

  // ---- PC scores and the kinship screen ------------------------------------
  Mat2D U = read_matrix(a.pref + ".eigvecs", false);
  if ((size_t)U.rows() != N) die(".eigvecs has " + std::to_string(U.rows()) + " rows, .fam has " + std::to_string(N));
  Eigen::Index k = a.npc > 0 ? a.npc : U.cols();
  if (k > U.cols()) die("-k is larger than the number of PCs in .eigvecs");
  Mat2D V(N, k + 1);
  V.leftCols(k) = U.leftCols(k);
  V.col(k).setOnes();
  const Mat2D VtV = V.transpose() * V;
  std::cout << "using " << k << " PC(s) + intercept"
            << (a.loo ? ", leave-one-out allele frequencies" : ", no leave-one-out") << "\n";

  Mat2D K = read_matrix(a.pref + ".kinship", true);
  if ((size_t)K.rows() != N || (size_t)K.cols() != N) die(".kinship is not " + std::to_string(N) + "x" + std::to_string(N));
  std::vector<std::pair<int, int>> pairs;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j)
      if (K(i, j) > a.min_kin) pairs.emplace_back((int)i, (int)j);
  std::cout << "pairs with kinship > " << a.min_kin << ": " << pairs.size() << " of " << N * (N - 1) / 2 << "\n";
  if (pairs.empty()) { std::cout << "nothing to do\n"; return 0; }

  // individuals appearing in at least one surviving pair; only their genotypes
  // are held, so memory is driven by the relatives, not by N
  std::vector<int> sub(N, -1);
  std::vector<int> members;
  for (auto& p : pairs) {
    if (sub[p.first] < 0)  { sub[p.first]  = (int)members.size(); members.push_back(p.first); }
    if (sub[p.second] < 0) { sub[p.second] = (int)members.size(); members.push_back(p.second); }
  }
  const size_t ni = members.size();
  std::cout << "individuals involved: " << ni << "\n";

  // ---- one streaming pass over the .bed ------------------------------------
  // R_s = V' g_s  (small, (k+1) x Mkeep) lets pi be formed on the fly later,
  // and -- unlike a stored beta -- lets the pair be removed from the fit per
  // pair. Genotypes are kept only for the involved individuals.
  Mat2D R(k + 1, Mkeep);
  std::vector<int8_t> Gsub(ni * Mkeep);
  {
    std::ifstream bed(a.plink + ".bed", std::ios::binary);
    if (!bed.is_open()) die("cannot open " + a.plink + ".bed");
    unsigned char magic[3];
    bed.read((char*)magic, 3);
    if (magic[0] != 0x6c || magic[1] != 0x1b || magic[2] != 0x01) die("not a SNP-major PLINK .bed");
    const size_t bpsnp = (N + 3) / 4;
    std::vector<unsigned char> buf(bpsnp);
    Mat1D g(N);
    size_t nmiss = 0;
    for (size_t j = 0; j < Mall; ++j) {
      bed.read((char*)buf.data(), bpsnp);
      if (!bed) die("unexpected end of .bed at site " + std::to_string(j));
      if (keep_idx[j] < 0) continue;
      const size_t c = keep_idx[j];
      for (size_t i = 0; i < N; ++i) {
        const double v = BED2GENO[(buf[i >> 2] >> ((i & 3) << 1)) & 3];
        if (v < 0) { ++nmiss; g(i) = 0.0; } else g(i) = v;
      }
      R.col(c).noalias() = V.transpose() * g;
      for (size_t t = 0; t < ni; ++t) Gsub[t * Mkeep + c] = (int8_t)g(members[t]);
    }
    if (nmiss) std::cerr << "warning: " << nmiss << " missing genotypes treated as 0; "
                         << "this tool assumes complete data\n";
  }

  // ---- per-pair EM ---------------------------------------------------------
  std::vector<std::array<double, 3>> out(pairs.size());
  std::vector<size_t> nsnp(pairs.size(), Mkeep);
  const bool a_loo = a.loo;
#pragma omp parallel
  {
    std::vector<double> lik(3 * Mkeep);
#pragma omp for schedule(dynamic)
    for (long t = 0; t < (long)pairs.size(); ++t) {
      const int ia = pairs[t].first, ib = pairs[t].second;
      const int8_t* ga = &Gsub[sub[ia] * Mkeep];
      const int8_t* gb = &Gsub[sub[ib] * Mkeep];
      const Mat1D va = V.row(ia).transpose(), vb = V.row(ib).transpose();
      // Leave-one-out: drop this pair's two rows from the regression that
      // produces pi, so pi_i and pi_j are not partly fitted to the relatedness
      // being measured. Both sides downdate by rank 2, so the cost is a
      // (k+1)x(k+1) solve per pair plus one extra axpy per site.
      Mat2D A = VtV;
      if (a_loo) { A.noalias() -= va * va.transpose(); A.noalias() -= vb * vb.transpose(); }
      const Mat2D Ainv = A.completeOrthogonalDecomposition().pseudoInverse();
      Mat1D rhs(k + 1), bs(k + 1);
      for (size_t s = 0; s < Mkeep; ++s) {
        rhs = R.col(s);
        if (a_loo) { rhs.noalias() -= va * (double)ga[s]; rhs.noalias() -= vb * (double)gb[s]; }
        bs.noalias() = Ainv * rhs;
        double pa = 0.5 * va.dot(bs);
        double pb = 0.5 * vb.dot(bs);
        pa = std::min(std::max(pa, 1e-3), 1.0 - 1e-3);
        pb = std::min(std::max(pb, 1e-3), 1.0 - 1e-3);
        const double ps = 0.5 * (pa + pb), qs = 1.0 - ps;
        const int A = ga[s], B = gb[s];
        lik[3 * s]     = dg(A, pa) * dg(B, pb);
        lik[3 * s + 1] = qs * aI(A, 0, pa) * aI(B, 0, pb) + ps * aI(A, 1, pa) * aI(B, 1, pb);
        lik[3 * s + 2] = (A == B) ? dg(A, ps) : 0.0;
      }
      double kk[3];
      em_ibd(lik, Mkeep, kk);
      out[t] = {kk[0], kk[1], kk[2]};
    }
  }

  // ---- output --------------------------------------------------------------
  const std::string fo = a.out + ".ibd";
  std::ofstream ofs(fo);
  if (!ofs.is_open()) die("cannot write " + fo);
  ofs << "ID1\tID2\tkin\tk0\tk1\tk2\tnsnp\n" << std::fixed << std::setprecision(6);
  for (size_t t = 0; t < pairs.size(); ++t)
    ofs << iid[pairs[t].first] << "\t" << iid[pairs[t].second] << "\t"
        << K(pairs[t].first, pairs[t].second) << "\t"
        << out[t][0] << "\t" << out[t][1] << "\t" << out[t][2] << "\t" << nsnp[t] << "\n";
  std::cout << "wrote " << pairs.size() << " pairs to " << fo << "\n";
  return 0;
}
