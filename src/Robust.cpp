/*******************************************************************************
 * @file        https://github.com/Zilong-Li/PCAone/src/Robust.cpp
 * @author      Anders Albrechtsen
 * Copyright (C) 2026. Use of this code is governed by the LICENSE file.
 *
 * PCA robust to close relatives (--robust), see Robust.hpp. A port of the
 * prototypes in scripts/relatepca/illustrate.py (cs_matrix, lr_kin,
 * lr_kin_pairs, whitened_cs, pcp_kin) and benchmark.py (king_robust).
 ******************************************************************************/
#include "Robust.hpp"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <tuple>

#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

#include "Halko.hpp"
#include "Kinship.hpp"
#include "Utils.hpp"

using MatB = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;
using Pair = std::tuple<int, int, double>;
// a block of genotype columns without copying (data->G.middleCols(...))
using GenoBlock = Eigen::Ref<const Mat2D>;

namespace {

struct Summaries {
  Mat2D A, HH, I0, C;  // raw Gram, het-het, IBS0, standardised GRM
  Mat2D As;            // dwg: SNP-standardised but uncentred Gram, sum_s g_s g_s' / (2 f_s (1 - f_s))
  Mat1D Dsum, nhet, gsum;
  double edge_sum = 0;    // sum over sites of Var(H_ij) * M^2 for an unrelated pair (noise edge)
  double edge_sum_s = 0;  // dwg: the same on the GRM scale
  uint64 M = 0, Ms = 0;   // sites; dwg: polymorphic sites in As
};

// per-site contribution to the noise variance of an off-diagonal entry of the CS
// matrix for an unrelated pair, under the pooled sample frequency f:
// E[g^2]^2 - (2f)^4 with E[g^2] = 2f(1-f) + 4f^2
double site_noise_var(const GenoBlock& X) {
  double v = 0;
  for (Eigen::Index j = 0; j < X.cols(); ++j) {
    const double f = X.col(j).mean();
    const double eg2 = 2.0 * f * (1.0 - f) + 4.0 * f * f;
    v += eg2 * eg2 - 16.0 * f * f * f * f;
  }
  return v;
}

// spectral edge of that noise: 2 sigma sqrt(N)
double noise_edge(double edge_sum, double M, Eigen::Index N) {
  return 2.0 * std::sqrt(edge_sum / (M * M)) * std::sqrt((double)N);
}

// add one block of genotypes on the 0..1 scale (N x b, x = g/2)
void accumulate(const GenoBlock& X, const Mat1D& f, bool grm, Summaries& s, bool scaled = false) {
  const Eigen::Index N = X.rows(), b = X.cols();
  Mat2D g = 2.0 * X;
  s.A.noalias() += g * g.transpose();
  s.Dsum += (g.array() * (2.0 - g.array())).rowwise().sum().matrix();
  Mat2D het = (X.array() == 0.5).cast<double>().matrix();
  Mat2D h0 = (X.array() == 0.0).cast<double>().matrix();
  Mat2D h2 = (X.array() == 1.0).cast<double>().matrix();
  s.HH.noalias() += het * het.transpose();
  Mat2D t = h0 * h2.transpose();
  s.I0 += t + t.transpose();
  s.nhet += het.rowwise().sum();
  s.edge_sum += site_noise_var(X);
  if (grm) {
    Mat2D Z(N, b);
    for (Eigen::Index j = 0; j < b; ++j) {
      const double sd = std::sqrt(2.0 * f(j) * (1.0 - f(j)));
      if (sd > VAR_TOL)
        Z.col(j) = (g.col(j).array() - 2.0 * f(j)) / sd;
      else
        Z.col(j).setZero();
    }
    s.C.noalias() += Z * Z.transpose();
  }
  if (scaled) {
    // dwg: standardised by 2f(1-f) but not centred; noise edge on that scale
    Mat2D Z(N, b);
    for (Eigen::Index j = 0; j < b; ++j) {
      const double fj = g.col(j).mean() / 2.0, w = 2.0 * fj * (1.0 - fj);
      if (w > VAR_TOL) {
        Z.col(j) = g.col(j) / std::sqrt(w);
        const double eg2 = 2.0 * fj * (1.0 - fj) + 4.0 * fj * fj;
        s.edge_sum_s += (eg2 * eg2 - 16.0 * fj * fj * fj * fj) / (w * w);
        ++s.Ms;
      } else {
        Z.col(j).setZero();
      }
    }
    s.As.noalias() += Z * Z.transpose();
    s.gsum += g.rowwise().sum();
  }
  s.M += b;
}

// top r eigenpairs of a symmetric matrix, largest first
void top_eig(const Mat2D& M, int r, Mat1D& w, Mat2D& V) {
  Eigen::SelfAdjointEigenSolver<Mat2D> es((M + M.transpose()) / 2.0);
  if (es.info() != Eigen::Success) cao.error("--robust: eigendecomposition failed");
  w = es.eigenvalues().tail(r).reverse();
  V = es.eigenvectors().rightCols(r).rowwise().reverse();
}

// one step of block subspace iteration: Y = Op(V), Rayleigh-Ritz on span(V),
// returns Ritz values (largest first), Ritz vectors and the next basis
// (the residual is checked on the first nchk Ritz pairs only: the extra
// oversampling vectors sit in the nearly flat noise spectrum and converge slowly)
template <class Op>
void ritz_step(const Op& op, Mat2D& V, Mat1D& theta, Mat2D& Vr, double& resid, Eigen::Index nchk,
               Mat1D* res = nullptr) {
  Mat2D Y = op(V);
  Mat2D B = V.transpose() * Y;
  Eigen::SelfAdjointEigenSolver<Mat2D> es((B + B.transpose()) / 2.0);
  theta = es.eigenvalues().reverse();
  Mat2D Z = es.eigenvectors().rowwise().reverse();
  Vr = V * Z;
  Mat2D Yr = Y * Z;
  resid = 0;
  for (Eigen::Index t = 0; t < std::min(nchk, Vr.cols()); ++t)
    resid = std::max(resid, (Yr.col(t) - theta(t) * Vr.col(t)).norm());
  resid /= std::max(std::abs(theta(0)), 1e-300);
  if (res) {
    res->resize(Vr.cols());
    for (Eigen::Index t = 0; t < Vr.cols(); ++t)
      (*res)(t) = (Yr.col(t) - theta(t) * Vr.col(t)).norm() / std::max(std::abs(theta(0)), 1e-300);
  }
  Eigen::HouseholderQR<Mat2D> qr(Yr);
  V = qr.householderQ() * Mat2D::Identity(Yr.rows(), Yr.cols());
}

// top r eigenpairs of a symmetric matrix by subspace iteration warm-started
// from V (N x b, b > r, kept between calls), so a sequence of slowly changing
// matrices costs O(N^2 b) per call instead of a full O(N^3) eigendecomposition;
// small N, or an iteration that does not converge, uses the full solver.
// The first nstrict pairs (default all r) are converged to 1e-12, the rest
// (eigenvectors in the flat noise spectrum, which converge slowly) to 1e-6
void top_eig_warm(const Mat2D& M, int r, Mat2D& V, Mat1D& w, Mat2D& U, int nstrict = -1) {
  const Eigen::Index n = M.rows();
  if (n <= 200 || V.cols() >= n) return top_eig(M, r, w, U);
  if (nstrict < 0 || nstrict > r) nstrict = r;
  auto op = [&](const Mat2D& X) { return Mat2D(M.selfadjointView<Eigen::Lower>() * X); };
  Mat1D theta, res;
  Mat2D Vr;
  double resid = 1;
  bool ok = false;
  for (int it = 0; it < 2000 && !ok; ++it) {
    ritz_step(op, V, theta, Vr, resid, nstrict, &res);
    ok = resid < 1e-12 && res.head(r).maxCoeff() < 1e-6;
  }
  if (!ok) return top_eig(M, r, w, U);
  w = theta.head(r);
  U = Vr.leftCols(r);
}

// whether eigenvalue r + 1 (index r) of op exceeds edge, iterating from V only
// until the answer is clear: Ritz values are lower bounds of the eigenvalues
// (theta_r > edge decides), and theta_r plus its residual norm below the edge
// (or convergence) decides the other way
template <class Op>
bool next_above_edge(const Op& op, Mat2D V, int r, double edge) {
  Mat1D theta, res;
  Mat2D Vr;
  double resid = 1;
  for (int it = 0; it < 2000; ++it) {
    ritz_step(op, V, theta, Vr, resid, r + 1, &res);
    if (theta(r) > edge) return true;
    if (theta(r) + res(r) * std::abs(theta(0)) < edge || resid < 1e-8) return false;
  }
  return false;
}

Mat2D rand_orth(Eigen::Index n, Eigen::Index b, uint seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  Mat2D R(n, b);
  for (Eigen::Index c = 0; c < b; ++c)
    for (Eigen::Index i = 0; i < n; ++i) R(i, c) = nd(rng);
  Eigen::HouseholderQR<Mat2D> qr(R);
  return qr.householderQ() * Mat2D::Identity(n, b);
}

// off-diagonal entries whose structure-adjusted kinship exceeds tau
MatB kin_select(const Mat2D& R, const Mat1D& v, double tau, const MatB& cand) {
  const Eigen::Index n = R.rows();
  MatB sel = MatB::Constant(n, n, false);
  for (Eigen::Index j = 0; j < n; ++j)
    for (Eigen::Index i = 0; i < n; ++i)
      if (i != j && cand(i, j)) {
        const double vv = std::max(v(i), 1e-12) * std::max(v(j), 1e-12);
        sel(i, j) = R(i, j) / (2.0 * std::sqrt(vv)) > tau;
      }
  return sel;
}

// fixed rank + kinship threshold, rank raised one step at a time (AltProj style)
// free_diag: the diagonal is imputed from the fit instead of trusting the
// observed one (then any diagonal shift of H, e.g. the CS matrix's -D, does
// not matter); off-diagonal entries are scaled to kinship with D
// edge > 0: the rank is raised only while the next eigenvalue of H - S exceeds
// the noise edge (rank is then the maximum); the rank used is returned in rank
int lr_kin(const Mat2D& H, int& rank, double tau, const Mat1D& D, const MatB& cand, Mat2D& L, Mat2D& S,
           bool free_diag = true, double edge = 0, int maxit = 500) {
  const Eigen::Index n = H.rows();
  S = Mat2D::Zero(n, n);
  int total = 0;
  Mat1D w;
  Mat2D V, Vb = rand_orth(n, std::min<Eigen::Index>(n, rank + 6), 1);
  for (int r = 1; r <= rank; ++r) {
    Mat2D Lold;
    for (int it = 0; it < maxit; ++it) {
      top_eig_warm(H - S, r, Vb, w, V);
      L.noalias() = V * w.asDiagonal() * V.transpose();
      Mat2D R = H - L;
      MatB sel = kin_select(R, D, tau, cand);
      S = sel.select(R, Mat2D::Zero(n, n));
      if (free_diag) S.diagonal() = R.diagonal();
      ++total;
      if (it > 0 && (L - Lold).cwiseAbs().maxCoeff() < 1e-10) break;
      Lold = L;
    }
    if (edge > 0 && r < rank) {
      bool above;
      if (n <= 200) {  // small N: full solver, as top_eig_warm
        Mat1D w2;
        Mat2D V2;
        top_eig(H - S, r + 1, w2, V2);
        above = w2(r) > edge;
      } else {
        const Mat2D HS = H - S;
        above = next_above_edge([&](const Mat2D& X) { return Mat2D(HS * X); }, Vb, r, edge);
      }
      if (!above) {
        rank = r;
        break;
      }
    }
  }
  return total;
}

// detected pairs and their structure-adjusted kinship
std::vector<Pair> lr_kin_pairs(const Mat2D& H, const Mat1D& D, const Mat2D& L, const Mat2D& S) {
  std::vector<Pair> pp;
  for (Eigen::Index j = 0; j < H.rows(); ++j)
    for (Eigen::Index i = 0; i < j; ++i)
      if (std::abs(S(i, j)) > 1e-12)
        pp.emplace_back(i, j, (H(i, j) - L(i, j)) / (2.0 * std::sqrt(std::max(D(i) * D(j), 1e-24))));
  return pp;
}

// structure PCs of a CS-type matrix: eigenvectors 2..k+1 (the first is the mean)
void cs_pcs(const Mat2D& M, int k, Mat1D& w, Mat2D& U) {
  Mat1D ww;
  Mat2D VV, V0 = rand_orth(M.rows(), std::min<Eigen::Index>(M.rows(), k + 7), 3);
  top_eig_warm(M, k + 1, V0, ww, VV);
  w = ww.tail(k);
  U = VV.rightCols(k);
}

// PCP (inexact ALM) of M with the diagonal unobserved and a kinship rule in
// place of the l1 shrinkage: S_ij = residual if its kinship > tau (and the
// pair passes the screen); the noise variance is the residual diagonal
// eigenvalue thresholding of a symmetric matrix: sum over |w| > t of
// sign(w)(|w| - t) v v'. full: complete eigendecomposition. Partial (IRAM,
// Spectra as in --svd 0): only the eigenpairs with |w| > t, largest magnitude
// first; if every computed one is above t, more are requested in the same
// call, so the result is exact up to the solver tolerance. sv carries the
// predicted count from one PCP iteration to the next.
struct SvtStats {
  long calls = 0, full = 0, max_nev = 0;
};
Mat2D eig_threshold(const Mat2D& A, double t, bool partial, int& sv, SvtStats& st) {
  const Eigen::Index n = A.rows();
  ++st.calls;
  if (partial) {
    Spectra::DenseSymMatProd<double> op(A);
    int nev = std::max(1, sv);
    while (nev < n / 2) {
      const int ncv = std::min<int>(n, std::max(2 * nev + 1, nev + 20));
      Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);
      eigs.init();
      eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10);
      if (eigs.info() != Spectra::CompInfo::Successful) break;
      const Mat1D w = eigs.eigenvalues();
      int kept = 0;
      for (Eigen::Index i = 0; i < w.size(); ++i) kept += std::abs(w(i)) > t;
      if (kept < nev) {
        st.max_nev = std::max<long>(st.max_nev, nev);
        sv = kept + 1;
        const Mat2D V = eigs.eigenvectors();
        Mat1D ws(w.size());
        for (Eigen::Index i = 0; i < w.size(); ++i)
          ws(i) = (w(i) > 0 ? 1.0 : -1.0) * std::max(std::abs(w(i)) - t, 0.0);
        return V * ws.asDiagonal() * V.transpose();
      }
      nev = std::min<int>(n, nev + std::max<int>(5, n / 20));  // all above t: ask for more
    }
  }
  ++st.full;
  Eigen::SelfAdjointEigenSolver<Mat2D> es(A);
  Mat1D w = es.eigenvalues();
  int kept = 0;
  for (Eigen::Index i = 0; i < n; ++i) {
    kept += std::abs(w(i)) > t;
    w(i) = (w(i) > 0 ? 1.0 : -1.0) * std::max(std::abs(w(i)) - t, 0.0);
  }
  sv = kept + 1;
  return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

Mat2D pcp_kin(const Mat2D& M, double tau, const MatB& cand, Mat2D& S, bool partial = false, int maxit = 1000,
              double tol = 1e-7) {
  const Eigen::Index n = M.rows();
  Mat2D Mo = M;
  Mo.diagonal().setZero();
  double norm2;
  if (partial && n > 50) {
    Spectra::DenseSymMatProd<double> op(Mo);
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 1, std::min<int>(n, 20));
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn, 1000, 1e-10);
    norm2 = std::abs(eigs.eigenvalues()(0));
  } else {
    Eigen::SelfAdjointEigenSolver<Mat2D> e0(Mo, Eigen::EigenvaluesOnly);
    norm2 = e0.eigenvalues().cwiseAbs().maxCoeff();
  }
  Mat2D Y = Mo / std::max(norm2, Mo.cwiseAbs().maxCoeff() * std::sqrt((double)n));
  double mu = 1.25 / norm2, rho = 1.5;
  const double mu_bar = mu * 1e7, nM = Mo.norm();
  S = Mat2D::Zero(n, n);
  Mat2D L(n, n);
  int sv = 10, it = 0;
  SvtStats st;
  for (; it < maxit; ++it) {
    Mat2D Aa = M - S + Y / mu;
    L = eig_threshold((Aa + Aa.transpose()) / 2.0, 1.0 / mu, partial, sv, st);
    Mat2D B = M - L + Y / mu;
    Mat1D noise = (M - L).diagonal();
    MatB sel = kin_select(B, noise, tau, cand);
    S = sel.select(B, Mat2D::Zero(n, n));
    S.diagonal() = (M - L).diagonal();
    Mat2D Z = M - L - S;
    Y += mu * Z;
    if (getenv("PCP_TRACE")) {
      int ns = 0;
      for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < i; ++j) ns += std::abs(S(i, j)) > 0;
      std::cerr << "PCP it " << it << " t=1/mu " << 1.0 / mu << " rank " << sv - 1 << " pairs " << ns << " relres "
                << Z.norm() / nM << "\n";
    }
    mu = std::min(mu * rho, mu_bar);
    if (Z.norm() / nM < tol) break;
  }
  cao.print(tick.date(), "robust PCA: PCP", it + 1, "iterations,", partial ? "partial (IRAM)" : "full",
            "eigen-thresholding; full decompositions", st.full, "of", st.calls, ", largest partial request", st.max_nev,
            ", final rank", sv - 1);
  return L;
}

// Soft-impute variant of the PCP (--pcp-edge): minimise
//   t ||L||_* + 1/2 || P_obs(M - L - S) ||^2
// with the diagonal unobserved and S on the KING-screened pairs chosen by the
// kinship rule, at a fixed threshold t = the noise edge of the off-diagonal
// GRM (2 sqrt(N/M) times the mean diagonal, 10% margin). Each iteration fills
// the diagonal with the current fit and soft-thresholds the eigenvalues, so L
// keeps the rank of the structure and the partial (IRAM) solve is cheap.
Mat2D pcp_kin_soft(const Mat2D& M, double tau, const MatB& cand, Mat2D& S, double t, bool partial,
                   int maxit = 2000, double tol = 1e-7) {
  const Eigen::Index n = M.rows();
  Mat2D L = Mat2D::Zero(n, n), Lprev;
  S = Mat2D::Zero(n, n);
  int sv = 10, it = 0;
  SvtStats st;
  for (; it < maxit; ++it) {
    Lprev = L;
    Mat2D F = M - S;
    F.diagonal() = L.diagonal();
    L = eig_threshold(F, t, partial, sv, st);
    Mat2D B = M - L;
    Mat1D noise = B.diagonal();
    MatB sel = kin_select(B, noise, tau, cand);
    S = sel.select(B, Mat2D::Zero(n, n));
    S.diagonal() = B.diagonal();
    if (it > 0 && (L - Lprev).norm() < tol * std::max(L.norm(), 1e-300)) break;
  }
  cao.print(tick.date(), "robust PCA: soft-impute PCP at the noise edge", t, ":", it + 1, "iterations,",
            partial ? "partial (IRAM)" : "full", "eigen-thresholding; full decompositions", st.full, "of", st.calls,
            ", final rank", sv - 1);
  return L;
}

// ---------------------------------------------------------------------------
// Operator engine (large N): nothing N x N is formed. Products with the
// genotypes are streamed one block of vectors at a time (in-core: G in memory;
// out-of-core: block reads). Eigenvectors come from warm-started block subspace
// iteration with Rayleigh-Ritz, so each iteration costs one pass over the data
// (single-vector IRAM would need a pass per Krylov vector).

// Z = (4/M) G G' V for a block of vectors, G on the 0..1 scale.
// Out-of-core: after the first pass PCAone's reader returns raw genotypes
// (frequencies known and params.center false), so nothing is added back here.
// genotypes as bit masks per individual (64 SNPs per word): heterozygous,
// homozygous 0 and homozygous 2; for KING-robust and Gram entries by popcount
struct PackedGeno {
  Eigen::Index N = 0, W = 0;
  std::vector<uint64_t> het, h0, h2;
  void init(Eigen::Index n, Eigen::Index nsnps) {
    N = n;
    W = (nsnps + 63) / 64;
    het.assign(N * W, 0);
    h0.assign(N * W, 0);
    h2.assign(N * W, 0);
  }
  // X: N x b on the 0..1 scale, sites first .. first + b - 1
  // (each thread owns a range of individuals; X is read down its columns)
  void add(const GenoBlock& X, Eigen::Index first, int nthreads) {
    const int nt = std::max(1, std::min<int>(nthreads, (int)(N / 64)));
    const Eigen::Index step = (N + nt - 1) / nt;
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; ++t) {
      const Eigen::Index r0 = t * step, r1 = std::min(N, r0 + step);
      for (Eigen::Index j = 0; j < X.cols(); ++j) {
        const Eigen::Index s = first + j, w = s / 64;
        const uint64_t bit = uint64_t(1) << (s % 64);
        for (Eigen::Index i = r0; i < r1; ++i) {
          const double x = X(i, j);
          if (x < 0.25)  // 0, 0.5, 1 up to rounding (out-of-core adds f back)
            h0[i * W + w] |= bit;
          else if (x < 0.75)
            het[i * W + w] |= bit;
          else
            h2[i * W + w] |= bit;
        }
      }
    }
  }
  int nhet(Eigen::Index i) const {
    int c = 0;
    for (Eigen::Index w = 0; w < W; ++w) c += __builtin_popcountll(het[i * W + w]);
    return c;
  }
  // sum over sites of g_i g_j, g = het + 2 h2
  double gram(Eigen::Index i, Eigen::Index j) const {
    uint64_t hh = 0, hd = 0, dd = 0;
    const uint64_t *ai = &het[i * W], *aj = &het[j * W], *bi = &h2[i * W], *bj = &h2[j * W];
    for (Eigen::Index w = 0; w < W; ++w) {
      hh += __builtin_popcountll(ai[w] & aj[w]);
      hd += __builtin_popcountll(ai[w] & bj[w]) + __builtin_popcountll(bi[w] & aj[w]);
      dd += __builtin_popcountll(bi[w] & bj[w]);
    }
    return double(hh) + 2.0 * double(hd) + 4.0 * double(dd);
  }
};

// KING-robust kinship (plink2 --make-king) of one pair; nh: heterozygous counts
inline double king_pair(const PackedGeno& P, const std::vector<int>& nh, Eigen::Index i, Eigen::Index j) {
  const Eigen::Index W = P.W;
  const uint64_t *hi = &P.het[i * W], *hj = &P.het[j * W];
  const uint64_t *ai = &P.h0[i * W], *aj = &P.h0[j * W], *ci = &P.h2[i * W], *cj = &P.h2[j * W];
  uint64_t hh = 0, i0 = 0;
  for (Eigen::Index w = 0; w < W; ++w) {
    hh += __builtin_popcountll(hi[w] & hj[w]);
    i0 += __builtin_popcountll(ai[w] & cj[w]) + __builtin_popcountll(ci[w] & aj[w]);
  }
  const double mn = std::max(std::min(nh[i], nh[j]), 1);
  return (double(hh) - 2.0 * double(i0)) / (2.0 * mn) + 0.5 - (nh[i] + nh[j]) / (4.0 * mn);
}

std::vector<int> het_counts(const PackedGeno& P) {
  std::vector<int> nh(P.N);
  for (Eigen::Index i = 0; i < P.N; ++i) nh[i] = P.nhet(i);
  return nh;
}

// KING-robust kinship for all pairs, in tiles of individuals over threads;
// returns the pairs above screen
std::vector<Pair> king_all_pairs(const PackedGeno& P, double screen, int nthreads) {
  const Eigen::Index N = P.N, T = 128, nb = (N + T - 1) / T;
  const std::vector<int> nh = het_counts(P);
  std::vector<std::vector<Pair>> found(std::max(1, nthreads));
#pragma omp parallel for num_threads(nthreads) schedule(dynamic) collapse(2)
  for (Eigen::Index bi = 0; bi < nb; ++bi)
    for (Eigen::Index bj = 0; bj < nb; ++bj) {
      if (bj < bi) continue;
      auto& out = found[omp_get_thread_num()];
      for (Eigen::Index i = bi * T; i < std::min(N, (bi + 1) * T); ++i)
        for (Eigen::Index j = std::max(i + 1, bj * T); j < std::min(N, (bj + 1) * T); ++j) {
          const double kin = king_pair(P, nh, i, j);
          if (kin > screen) out.emplace_back(i, j, kin);
        }
    }
  std::vector<Pair> pairs;
  for (auto& f : found) pairs.insert(pairs.end(), f.begin(), f.end());
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

// Count sketch of the standardised genotypes for the relative search (very
// large N, scripts/relatepca/relfind_bits.py): site s is added with sign
// sigma(s) to column h(s), both from a hash of the site index, so one pass
// costs N x M. For a pair the correlation of their sketch rows estimates the
// genotype correlation (about 2 phi for relatives, 0 +- 1/sqrt(dim) otherwise).
inline uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct RelSketch {
  Eigen::Index N = 0, dim = 0;
  Eigen::MatrixXf Y;  // N x dim, later dim x N with unit columns (see finish)
  void init(Eigen::Index n, Eigen::Index d) {
    N = n;
    dim = d;
    Y = Eigen::MatrixXf::Zero(N, dim);
  }
  // X: N x b on the 0..1 scale, sites first .. first + b - 1
  void add(const GenoBlock& X, Eigen::Index first, int nthreads) {
    const Eigen::Index b = X.cols();
    std::vector<double> f(b), sc(b);
    std::vector<Eigen::Index> col(b);
    for (Eigen::Index j = 0; j < b; ++j) {
      f[j] = X.col(j).mean();
      const double v = 2.0 * f[j] * (1.0 - f[j]);
      const uint64_t h = mix64(uint64_t(first + j));
      col[j] = Eigen::Index((h >> 1) % uint64_t(dim));
      sc[j] = v > 1e-12 ? ((h & 1) ? 2.0 : -2.0) / std::sqrt(v) : 0.0;  // g = 2x
    }
    const int nt = std::max(1, std::min<int>(nthreads, (int)(N / 64)));
    const Eigen::Index step = (N + nt - 1) / nt;
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; ++t) {
      const Eigen::Index r0 = t * step, r1 = std::min(N, r0 + step);
      for (Eigen::Index j = 0; j < b; ++j) {
        if (sc[j] == 0.0) continue;
        float* y = Y.col(col[j]).data();
        for (Eigen::Index i = r0; i < r1; ++i) y[i] += float(sc[j] * (X(i, j) - f[j]));
      }
    }
  }
  void finish() {
    Eigen::MatrixXf Yt = Y.transpose();
    Y.resize(0, 0);
    for (Eigen::Index i = 0; i < N; ++i) {
      const float nrm = Yt.col(i).norm();
      if (nrm > 0) Yt.col(i) /= nrm;
    }
    Y.swap(Yt);
  }
};

// the m most similar individuals of each individual in rows r0 .. r1 - 1
// (blocked products over tiles of columns; Y: dim x N with unit columns)
void top_neighbours(const Eigen::MatrixXf& Y, Eigen::Index r0, Eigen::Index r1, int m,
                    std::vector<std::vector<int>>& nbr) {
  const Eigen::Index N = Y.cols(), nb = r1 - r0, CT = 4096;
  std::vector<std::vector<std::pair<float, int>>> best(nb);
  for (Eigen::Index c0 = 0; c0 < N; c0 += CT) {
    const Eigen::Index nc = std::min(CT, N - c0);
    const Eigen::MatrixXf S = Y.middleCols(r0, nb).transpose() * Y.middleCols(c0, nc);
    for (Eigen::Index a = 0; a < nb; ++a) {
      auto& h = best[a];  // min-heap of the m largest
      for (Eigen::Index c = 0; c < nc; ++c) {
        const Eigen::Index j = c0 + c;
        if (j == r0 + a) continue;
        const float v = S(a, c);
        if ((int)h.size() < m) {
          h.emplace_back(v, (int)j);
          std::push_heap(h.begin(), h.end(), std::greater<>());
        } else if (v > h.front().first) {
          std::pop_heap(h.begin(), h.end(), std::greater<>());
          h.back() = {v, (int)j};
          std::push_heap(h.begin(), h.end(), std::greater<>());
        }
      }
    }
  }
  for (Eigen::Index a = 0; a < nb; ++a) {
    std::sort(best[a].begin(), best[a].end(), std::greater<>());
    nbr[r0 + a].clear();
    for (const auto& [v, j] : best[a]) nbr[r0 + a].push_back(j);
  }
}

// candidate pairs from the sketch neighbours, verified with KING-robust. If all
// m neighbours of an individual are above screen (a large family), its search
// is extended to 2m, 4m, ... until one is not. Returns the pairs above screen.
std::vector<Pair> king_sketch_pairs(const PackedGeno& P, const RelSketch& sk, int m, double screen, int nthreads,
                                    uint64& nchecked, uint64& nextended) {
  const Eigen::Index N = P.N;
  const std::vector<int> nh = het_counts(P);
  m = (int)std::min<Eigen::Index>(m, N - 1);
  std::vector<std::vector<int>> nbr(N);
  const Eigen::Index B = 64;
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  for (Eigen::Index r0 = 0; r0 < N; r0 += B) top_neighbours(sk.Y, r0, std::min(N, r0 + B), m, nbr);
  std::map<std::pair<int, int>, double> kin;  // checked pairs
  auto check = [&](const std::vector<std::pair<int, int>>& todo) {
    std::vector<double> k(todo.size());
#pragma omp parallel for num_threads(nthreads)
    for (size_t c = 0; c < todo.size(); ++c) k[c] = king_pair(P, nh, todo[c].first, todo[c].second);
    for (size_t c = 0; c < todo.size(); ++c) kin[todo[c]] = k[c];
  };
  auto key = [](int i, int j) { return std::make_pair(std::min(i, j), std::max(i, j)); };
  auto unchecked = [&](const std::vector<Eigen::Index>& who) {
    std::vector<std::pair<int, int>> todo;
    for (auto i : who)
      for (int j : nbr[i])
        if (!kin.count(key(i, j))) todo.push_back(key(i, j));
    std::sort(todo.begin(), todo.end());
    todo.erase(std::unique(todo.begin(), todo.end()), todo.end());
    return todo;
  };
  std::vector<Eigen::Index> who(N);
  std::iota(who.begin(), who.end(), 0);
  nextended = 0;
  for (int mm = m;; mm *= 2) {
    check(unchecked(who));
    // individuals whose neighbours are all relatives: search further
    std::vector<Eigen::Index> full;
    for (auto i : who) {
      bool all = (int)nbr[i].size() == mm;
      for (int j : nbr[i]) all = all && kin[key(i, j)] > screen;
      if (all) full.push_back(i);
    }
    if (full.empty() || mm >= N - 1) break;
    nextended += full.size();
    const int m2 = (int)std::min<Eigen::Index>(2 * mm, N - 1);
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for (size_t c = 0; c < full.size(); ++c) top_neighbours(sk.Y, full[c], full[c] + 1, m2, nbr);
    who = full;
  }
  nchecked = kin.size();
  std::vector<Pair> pairs;
  for (const auto& [ij, k] : kin)
    if (k > screen) pairs.emplace_back(ij.first, ij.second, k);
  return pairs;  // sorted by (i, j)
}

// Z += G (G' V), with the SNPs split over threads (Eigen parallelises a product
// over the columns of its result, i.e. at most cols / 4 threads for the few
// vectors used here); each thread sums into its own N x b block
void add_GGtV(const GenoBlock& G, const Mat2D& V, Mat2D& Z, int nthreads) {
  const Eigen::Index M = G.cols();
  const int nt = std::max(1, std::min<int>(nthreads, (int)(M / 256)));
  if (nt == 1) {
    Mat2D T = G.transpose() * V;
    Z.noalias() += G * T;
    return;
  }
  const Eigen::Index nchunk = 4 * nt, step = (M + nchunk - 1) / nchunk;
  std::vector<Mat2D> Zt(nt, Mat2D::Zero(Z.rows(), Z.cols()));
#pragma omp parallel for num_threads(nt) schedule(dynamic)
  for (Eigen::Index c = 0; c < nchunk; ++c) {
    const Eigen::Index s0 = c * step, len = std::min(step, M - s0);
    if (len <= 0) continue;
    Mat2D T = G.middleCols(s0, len).transpose() * V;
    Zt[omp_get_thread_num()].noalias() += G.middleCols(s0, len) * T;
  }
  for (const auto& z : Zt) Z += z;
}

struct GenoProduct {
  Data* data;
  const Param& params;
  double M;
  mutable uint64 passes = 0;
  mutable double secs = 0;  // time spent in the products
  Mat2D apply(const Mat2D& V) const {
    ++passes;
    const auto t0 = std::chrono::steady_clock::now();
    Mat2D Z = Mat2D::Zero(V.rows(), V.cols());
    if (!params.out_of_core) {
      add_GGtV(data->G, V, Z, params.threads);
    } else {
      data->check_file_offset_first_var();
      for (uint bi = 0; bi < data->nblocks; ++bi) {
        data->read_block_initial(data->start[bi], data->stop[bi], false);
        add_GGtV(data->G, V, Z, params.threads);
      }
    }
    Z *= 4.0 / M;
    secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return Z;
  }
};

// block-diagonal Sigma^{-1/2} (or Sigma^{1/2}): a scalar for singletons, a dense block per family
struct BlockDiag {
  Mat1D scalar;              // used for samples not in a family
  std::vector<Int1D> fam;    // family members
  std::vector<Mat2D> block;  // per family
  Mat2D apply(const Mat2D& X) const {
    Mat2D Y = scalar.asDiagonal() * X;
    for (size_t f = 0; f < fam.size(); ++f) {
      Mat2D Xf = X(fam[f], Eigen::all);
      Y(fam[f], Eigen::all) = block[f] * Xf;
    }
    return Y;
  }
  // the same in place (rows of a block of genotypes)
  void apply_inplace(Mat2D& X) const {
    std::vector<Mat2D> Xf(fam.size());  // family rows before scaling (scalar is 0 there)
    for (size_t f = 0; f < fam.size(); ++f) Xf[f] = X(fam[f], Eigen::all);
    X = scalar.asDiagonal() * X;
    for (size_t f = 0; f < fam.size(); ++f) X(fam[f], Eigen::all) = block[f] * Xf[f];
  }
};

// Sigma^{-1/2} and Sigma^{1/2} with Sigma = v^1/2 (I + 2 Psi) v^1/2, families
// from the pairs; the eigenvalue floor matches the dense version
void build_whitener(const Mat1D& v, const std::vector<Pair>& pairs, BlockDiag& Wm, BlockDiag& Wh) {
  const Eigen::Index n = v.size();
  Mat1D d = v.cwiseMax(1e-12).cwiseSqrt();
  const double floor = 0.1 * d.minCoeff() * d.minCoeff();
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> root = [&](int i) { return parent[i] == i ? i : parent[i] = root(parent[i]); };
  for (const auto& [i, j, phi] : pairs) parent[root(i)] = root(j);
  std::map<int, Int1D> comp;
  for (int i = 0; i < n; ++i) comp[root(i)].push_back(i);
  Wm.scalar = Wh.scalar = Mat1D::Zero(n);
  std::map<std::pair<int, int>, double> phi_of;
  for (const auto& [i, j, phi] : pairs) phi_of[{i, j}] = phi;
  for (auto& kv : comp) {
    const Int1D& f = kv.second;
    if (f.size() == 1) {
      const double sig = std::max(v(f[0]), floor);
      Wm.scalar(f[0]) = 1.0 / std::sqrt(sig);
      Wh.scalar(f[0]) = std::sqrt(sig);
      continue;
    }
    const int m = f.size();
    Mat2D Sig(m, m);
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b) {
        double r = (a == b) ? 1.0 : 0.0;
        auto it = phi_of.find({std::min(f[a], f[b]), std::max(f[a], f[b])});
        if (a != b && it != phi_of.end()) r = 2.0 * it->second;
        Sig(a, b) = d(f[a]) * r * d(f[b]);
      }
    Eigen::SelfAdjointEigenSolver<Mat2D> es(Sig);
    Mat1D ev = es.eigenvalues().cwiseMax(floor);
    const Mat2D& Q = es.eigenvectors();
    Wm.fam.push_back(f);
    Wh.fam.push_back(f);
    Wm.block.push_back(Q * ev.cwiseSqrt().cwiseInverse().asDiagonal() * Q.transpose());
    Wh.block.push_back(Q * ev.cwiseSqrt().asDiagonal() * Q.transpose());
  }
}

// CS whitening: eigenvectors of Sigma^-1/2 (A/M) Sigma^-1/2, mapped back by
// Sigma^1/2, with Sigma = D^1/2 (I + 2 Psi) D^1/2; the first PC (mean) dropped
// noise: per-individual noise variance (D for CS whitening; the observed
// diagonal minus the fitted structure for the free-diagonal version)
// nstrict: number of leading pairs converged strictly (the structure rank)
// centre (dwg): instead of dropping the first PC, the whitened matrix is
// centred by projecting out u = Sigma^-1/2 1 (the mean direction in the
// whitened space; the same as centring every SNP by its 1/v-weighted mean).
// The whitened noise I becomes I - uu' (still flat), so the PCs are the top k.
void whitened_cs(const Mat2D& AM, const Mat1D& noise, const std::vector<Pair>& pairs, int k, Mat1D& w, Mat2D& U,
                 int nstrict = -1, bool centre = false) {
  const Eigen::Index n = AM.rows();
  BlockDiag Wm, Wh;
  build_whitener(noise, pairs, Wm, Wh);
  Mat2D B = Wm.apply(Mat2D(Wm.apply(AM).transpose()));
  Mat1D ww;
  Mat2D VV, V0 = rand_orth(n, std::min<Eigen::Index>(n, k + 7), 2);
  if (centre) {
    Mat1D u = Wm.apply(Mat2D::Ones(n, 1)).col(0);
    u.normalize();
    const Mat1D Bu = B * u;
    B -= Bu * u.transpose() + u * Bu.transpose() - (u.dot(Bu)) * u * u.transpose();  // (I - uu') B (I - uu')
    top_eig_warm(B, k, V0, ww, VV, nstrict > 0 ? nstrict - 1 : nstrict);
    w = ww;
    U = Wh.apply(VV);
    return;
  }
  top_eig_warm(B, k + 1, V0, ww, VV, nstrict);
  w = ww.tail(k);
  U = Wh.apply(VV).rightCols(k);
}

void write_matrix(const std::string& fn, const Mat2D& M) {
  std::ofstream ofs(fn);
  if (!ofs.is_open()) cao.error("can not open " + fn);
  Eigen::IOFormat fmt(6, Eigen::DontAlignCols, "\t", "\n");
  ofs << M.format(fmt) << "\n";
}

}  // namespace

// ---- final relatedness of the related pairs, given the final PCs -----------
// The kinship used inside the methods only decides which pairs are related and
// how much they are down-weighted. The reported relatedness is estimated
// afterwards from individual allele frequencies (IAF) implied by the final PCs,
// in one more pass over the genotypes:
//   k0 k1 k2  IBD sharing probabilities by maximum likelihood with pi_i, pi_j
//             from the PCs (the likelihood of
//             pcaone-ibd, src/PcaoneIbd.cpp), but with the pair's
//             whole family (connected through related pairs, up to
//             max_family members) left out of the IAF fit, not only the pair:
//             at small N the other members otherwise absorb part of the shared
//             signal (4 sibs among 15 of a population: k0 0.32 instead of 0.25)
//   kinship   k1/4 + k2/2 from the same fit
//   kin_ea    half the correlation of residuals, the evalAdmix projection
//             estimator of --evaladmix (src/EvalAdmix.cpp), for these pairs only;
//             no leave-out, so biased low for large families at small N
struct FinalRel {
  double kin = NAN, kin_ea = NAN, k0 = NAN, k1 = NAN, k2 = NAN;
};

static inline double hwe_prob(int g, double p) {
  const double q = 1.0 - p;
  return g == 0 ? q * q : (g == 1 ? 2.0 * p * q : p * p);
}
static inline double other_allele(int g, int shared, double p) {
  const int d = g - shared;
  return d == 0 ? 1.0 - p : (d == 1 ? p : 0.0);
}
// Maximum likelihood (k0, k1, k2) on the simplex; lik: 3 per site. The
// log-likelihood sum_s log(k . l_s) is concave, so a projected Newton method
// with an active set (components fixed at 0) converges in a few passes. Plain
// EM, as in pcaone-ibd, needs hundreds of passes when the estimate is on the
// boundary (parent-offspring: k0 = 0; MZ: k0 = k1 = 0).
// log-likelihood (if want_ll), gradient g and Hessian H (if not null)
static double ibd_loglik(const std::vector<double>& lik, size_t ns, const double* k, double* g, double* H,
                         bool want_ll = true) {
  double ll = 0, g0 = 0, g1 = 0, g2 = 0, h00 = 0, h01 = 0, h02 = 0, h11 = 0, h12 = 0, h22 = 0;
  for (size_t s = 0; s < ns; ++s) {
    const double* l = &lik[3 * s];
    const double t = k[0] * l[0] + k[1] * l[1] + k[2] * l[2];
    if (!(t > 0)) {
      if (l[0] + l[1] + l[2] > 0) return -INFINITY;  // impossible under k
      continue;
    }
    if (want_ll) ll += std::log(t);
    if (g) {
      const double it = 1.0 / t, w0 = l[0] * it, w1 = l[1] * it, w2 = l[2] * it;
      g0 += w0;
      g1 += w1;
      g2 += w2;
      h00 += w0 * w0;
      h01 += w0 * w1;
      h02 += w0 * w2;
      h11 += w1 * w1;
      h12 += w1 * w2;
      h22 += w2 * w2;
    }
  }
  if (g) {
    g[0] = g0;
    g[1] = g1;
    g[2] = g2;
    const double h[9] = {h00, h01, h02, h01, h11, h12, h02, h12, h22};
    if (H)
      for (int a = 0; a < 9; ++a) H[a] = -h[a];
  }
  return ll;
}

// k: start on input, estimate on output
static void ml_ibd(const std::vector<double>& lik, size_t ns, double* k) {
  bool act[3];  // components fixed at 0
  for (int a = 0; a < 3; ++a) act[a] = k[a] <= 0;
  double g[3], H[9];
  double ll = ibd_loglik(lik, ns, k, g, H);
  for (int it = 0; it < 200; ++it) {
    int f[3], nf = 0;
    for (int a = 0; a < 3; ++a)
      if (!act[a]) f[nf++] = a;
    // Newton direction on the face: (-Hff) d = g_f - mu 1, sum d = 0
    Eigen::Matrix3d Hf = Eigen::Matrix3d::Identity();
    Eigen::Vector3d gf = Eigen::Vector3d::Zero(), one = Eigen::Vector3d::Zero();
    for (int a = 0; a < nf; ++a) {
      gf(a) = g[f[a]];
      one(a) = 1;
      for (int b = 0; b < nf; ++b) Hf(a, b) = -H[3 * f[a] + f[b]];
      Hf(a, a) += 1e-12 * (1 + std::fabs(Hf(a, a)));
    }
    const Eigen::Matrix3d Hi = Hf.inverse();
    const double mu = one.dot(Hi * gf) / std::max(one.dot(Hi * one), 1e-300);
    const Eigen::Vector3d dn = Hi * (gf - mu * one);
    double d[3] = {0, 0, 0}, dsz = 0;
    bool bad = false;  // Newton would push a component sitting at 0 below 0
    for (int a = 0; a < nf; ++a) {
      d[f[a]] = dn(a);
      dsz += std::fabs(dn(a));
      if (k[f[a]] <= 0 && dn(a) < 0) bad = true;
    }
    if (bad) {
      // ascent direction of the face that keeps zero components feasible:
      // d_a = (g_a - lam) - k_a sum_b (g_b - lam), lam = sum_b g_b k_b
      double lam = 0, sg = 0;
      for (int a = 0; a < nf; ++a) lam += g[f[a]] * k[f[a]];
      for (int a = 0; a < nf; ++a) sg += g[f[a]] - lam;
      dsz = 0;
      for (int a = 0; a < nf; ++a) {
        d[f[a]] = (g[f[a]] - lam - k[f[a]] * sg) / std::max(lam, 1.0);
        dsz += std::fabs(d[f[a]]);
      }
    }
    bool face_done = dsz < 1e-11;
    if (!face_done) {
      double tmax = 1.0;
      int hit = -1;
      for (int a = 0; a < 3; ++a)
        if (d[a] < 0 && k[a] + tmax * d[a] < 0) {
          tmax = -k[a] / d[a];
          hit = a;
        }
      double t = tmax, kn[3], lln = -INFINITY;
      for (int bt = 0; bt < 50; ++bt, t *= 0.5) {
        for (int a = 0; a < 3; ++a) kn[a] = std::max(0.0, k[a] + t * d[a]);
        if (t == tmax && hit >= 0) kn[hit] = 0;
        const double sum = kn[0] + kn[1] + kn[2];
        for (int a = 0; a < 3; ++a) kn[a] /= sum;
        lln = ibd_loglik(lik, ns, kn, nullptr, nullptr);
        if (lln >= ll) break;
      }
      if (!(lln >= ll)) {
        face_done = true;
      } else {
        const double change = std::fabs(kn[0] - k[0]) + std::fabs(kn[1] - k[1]) + std::fabs(kn[2] - k[2]);
        for (int a = 0; a < 3; ++a) {
          k[a] = kn[a];
          if (k[a] <= 0) act[a] = true;
        }
        ll = lln;
        ibd_loglik(lik, ns, k, g, H, false);
        if (change > 1e-12) continue;
        face_done = true;
      }
    }
    // optimum of the face: release the active component that violates the
    // KKT condition most (g_a > lam: moving mass to it raises the likelihood)
    double lam = 0;
    for (int a = 0; a < 3; ++a) lam += g[a] * k[a];
    int rel = -1;
    double worst = 1e-9 * std::max(lam, 1.0);
    for (int a = 0; a < 3; ++a)
      if (act[a] && g[a] - lam > worst) {
        worst = g[a] - lam;
        rel = a;
      }
    if (rel < 0) break;
    act[rel] = false;
  }
}

static std::vector<FinalRel> final_relatedness(Data* data, const Param& params, const Mat2D& U,
                                               const std::vector<Pair>& pairs) {
  const Eigen::Index N = data->nsamples, k1 = U.cols() + 1, np = pairs.size();
  std::vector<FinalRel> out(np);
  if (np == 0) return out;
  tick.clock();
  Mat2D V(N, k1);
  V.leftCols(k1 - 1) = U;
  V.col(k1 - 1).setOnes();
  // genotypes are kept only for the individuals in a pair (memory follows the
  // relatives, not N)
  std::vector<int> sub(N, -1), members;
  for (const auto& [i, j, phi] : pairs)
    for (int x : {i, j})
      if (sub[x] < 0) {
        sub[x] = members.size();
        members.push_back(x);
      }
  const Eigen::Index ni = members.size(), M = data->nsnps;
  Mat2D R(k1, M);  // V' g per site
  std::vector<int8_t> Gsub(ni * M);
  Mat2D AV = Mat2D::Zero(N, k1);
  Mat1D Adiag = Mat1D::Zero(N), bsum = Mat1D::Zero(N), dsum = Mat1D::Zero(N), Ap = Mat1D::Zero(np);
  Eigen::Index s0 = 0;
  // X: N x b on the 0..1 scale, g = 2X; the sums are rescaled after the pass
  auto add_block = [&](const GenoBlock& X) {
    const Eigen::Index b = X.cols();
    add_GGtV(X, V, AV, params.threads);
    // per-sample sums, V' g per site, pair products and the members' genotypes
    // over small chunks of sites (transposed: an individual's sites contiguous)
    const Eigen::Index cs = 256, nch = (b + cs - 1) / cs;
    const int nt = std::max(1, (int)params.threads);
    std::vector<Mat1D> Apt(nt, Mat1D::Zero(np)), Adt(nt, Mat1D::Zero(N)), bt(nt, Mat1D::Zero(N)),
        dt(nt, Mat1D::Zero(N));
#pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (Eigen::Index ch = 0; ch < nch; ++ch) {
      const Eigen::Index c0 = ch * cs, len = std::min(cs, b - c0);
      const auto Xc = X.middleCols(c0, len);
      const int th = omp_get_thread_num();
      R.middleCols(s0 + c0, len).noalias() = 2.0 * (V.transpose() * Xc);
      Adt[th] += Xc.array().square().rowwise().sum().matrix();
      bt[th] += Xc.rowwise().sum();
      dt[th] += (Xc.array() * (1.0 - Xc.array())).rowwise().sum().matrix();
      // the members only, transposed: len x ni
      Mat2D xt(len, ni);
      for (Eigen::Index j = 0; j < len; ++j)
        for (Eigen::Index t = 0; t < ni; ++t) xt(j, t) = Xc(members[t], j);
      Mat1D& acc = Apt[th];
      for (Eigen::Index c = 0; c < np; ++c)
        acc(c) += xt.col(sub[std::get<0>(pairs[c])]).dot(xt.col(sub[std::get<1>(pairs[c])]));
      for (Eigen::Index t = 0; t < ni; ++t) {
        const double* gc = xt.col(t).data();
        int8_t* dst = &Gsub[t * M + s0 + c0];
        for (Eigen::Index j = 0; j < len; ++j) dst[j] = (int8_t)(2.0 * gc[j] + 0.5);
      }
    }
    for (int t = 0; t < nt; ++t) {
      Ap += Apt[t];
      Adiag += Adt[t];
      bsum += bt[t];
      dsum += dt[t];
    }
    s0 += b;
  };
  if (!params.out_of_core) {
    for (Eigen::Index c = 0; c < data->G.cols(); c += 4096)
      add_block(data->G.middleCols(c, std::min<Eigen::Index>(4096, data->G.cols() - c)));
  } else {
    // allele frequencies are known after the first pass, so the reader returns
    // raw genotypes (as in GenoProduct)
    data->check_file_offset_first_var();
    for (uint bi = 0; bi < data->nblocks; ++bi) {
      data->read_block_initial(data->start[bi], data->stop[bi], false);
      add_block(data->G);
    }
  }
  AV *= 4.0;
  Adiag *= 4.0;
  bsum *= 2.0;
  dsum *= 4.0;
  Ap *= 4.0;
  const double Md = (double)M;
  const double t_pass = tick.reltime();
  tick.clock();
  // evalAdmix: corres = cov2cor((I-P) C (I-P)) - cov2cor((I-P) D (I-P)), with
  // C = G G' - M gbar gbar', D = diag(mean heterozygosity), P the projection on V
  const Mat1D bm = bsum / Md, dm = dsum / Md;
  const Mat2D Pinv = (V.transpose() * V).completeOrthogonalDecomposition().pseudoInverse();
  const Mat2D CV = AV - Md * bm * (bm.transpose() * V);  // C V
  const Mat2D VCV = V.transpose() * CV;
  const Mat2D VDV = V.transpose() * dm.asDiagonal() * V;
  const Mat2D VP = V * Pinv;  // rows v_i' Pinv
  auto cw = [&](Eigen::Index i, Eigen::Index j, double Cij) {
    return Cij - VP.row(i).dot(CV.row(j)) - CV.row(i).dot(VP.row(j)) + VP.row(i) * VCV * VP.row(j).transpose();
  };
  auto ew = [&](Eigen::Index i, Eigen::Index j) {
    const double Pij = VP.row(i).dot(V.row(j));
    return (i == j ? dm(i) : 0.0) - Pij * dm(j) - dm(i) * Pij + VP.row(i) * VDV * VP.row(j).transpose();
  };
  for (Eigen::Index c = 0; c < np; ++c) {
    const Eigen::Index i = std::get<0>(pairs[c]), j = std::get<1>(pairs[c]);
    const double bij = cw(i, j, Ap(c) - Md * bm(i) * bm(j)) /
                       std::sqrt(std::max(cw(i, i, Adiag(i) - Md * bm(i) * bm(i)) *
                                              cw(j, j, Adiag(j) - Md * bm(j) * bm(j)),
                                          1e-300));
    const double cij = ew(i, j) / std::sqrt(std::max(ew(i, i) * ew(j, j), 1e-300));
    out[c].kin_ea = 0.5 * std::min(1.0, std::max(-1.0, bij - cij));
  }
  const double t_ea = tick.reltime();
  tick.clock();
  // families: connected components of the related pairs (union-find on the
  // member index); a family larger than max_family falls back to the pair
  const Eigen::Index max_family = 100;
  std::vector<int> root(ni);
  std::iota(root.begin(), root.end(), 0);
  std::function<int(int)> find = [&](int x) { return root[x] == x ? x : root[x] = find(root[x]); };
  for (const auto& [i, j, phi] : pairs) root[find(sub[i])] = find(sub[j]);
  std::map<int, std::vector<int>> family;  // root -> member indices (into members)
  std::vector<int> fam_of(ni);              // no path compression inside the parallel loop
  for (Eigen::Index t = 0; t < ni; ++t) family[fam_of[t] = find(t)].push_back(t);
  // IBD sharing by EM (pcaone-ibd) with leave-the-family-out IAF
  const Mat2D VtV = V.transpose() * V;
#pragma omp parallel num_threads(params.threads)
  {
    std::vector<double> lik(3 * M);
#pragma omp for schedule(dynamic)
    for (Eigen::Index c = 0; c < np; ++c) {
      const int ia = std::get<0>(pairs[c]), ib = std::get<1>(pairs[c]);
      const int8_t *ga = &Gsub[sub[ia] * M], *gb = &Gsub[sub[ib] * M];
      const Mat1D va = V.row(ia).transpose(), vb = V.row(ib).transpose();
      std::vector<int> out_fit = family.at(fam_of[sub[ia]]);
      if ((Eigen::Index)out_fit.size() > max_family) out_fit = {sub[ia], sub[ib]};
      Mat2D A = VtV;
      for (int t : out_fit) A.noalias() -= V.row(members[t]).transpose() * V.row(members[t]);
      const Mat2D Ainv = A.completeOrthogonalDecomposition().pseudoInverse();
      // pi_a = 0.5 va' Ainv (R_s - sum_l v_l g_ls) over the left-out members l
      const Mat1D wa = Ainv * va, wb = Ainv * vb;
      const Mat1D ra = R.transpose() * wa, rb = R.transpose() * wb;  // M
      const Eigen::Index nf = out_fit.size();
      std::vector<double> ca(nf), cb(nf);
      std::vector<const int8_t*> gl(nf);
      for (Eigen::Index l = 0; l < nf; ++l) {
        ca[l] = wa.dot(V.row(members[out_fit[l]]));
        cb[l] = wb.dot(V.row(members[out_fit[l]]));
        gl[l] = &Gsub[out_fit[l] * M];
      }
      double ibs0 = 0, ibs0_exp = 0;
      for (Eigen::Index s = 0; s < M; ++s) {
        const int A1 = ga[s], B1 = gb[s];
        double pa = 0.5 * ra(s), pb = 0.5 * rb(s);
        for (Eigen::Index l = 0; l < nf; ++l) {
          pa -= 0.5 * ca[l] * gl[l][s];
          pb -= 0.5 * cb[l] * gl[l][s];
        }
        pa = std::min(std::max(pa, 1e-3), 1.0 - 1e-3);
        pb = std::min(std::max(pb, 1e-3), 1.0 - 1e-3);
        const double ps = 0.5 * (pa + pb), qs = 1.0 - ps;
        lik[3 * s] = hwe_prob(A1, pa) * hwe_prob(B1, pb);
        lik[3 * s + 1] = qs * other_allele(A1, 0, pa) * other_allele(B1, 0, pb) +
                         ps * other_allele(A1, 1, pa) * other_allele(B1, 1, pb);
        lik[3 * s + 2] = (A1 == B1) ? hwe_prob(A1, ps) : 0.0;
        ibs0 += (A1 + B1 == 2 && A1 != 1);
        ibs0_exp += pa * pa * (1 - pb) * (1 - pb) + (1 - pa) * (1 - pa) * pb * pb;
      }
      // start: k0 from opposite homozygotes, k2 from the detection kinship
      // (phi = k1/4 + k2/2), moved into the interior
      double kk[3];
      kk[0] = std::min(1.0, std::max(0.0, ibs0 / std::max(ibs0_exp, 1e-12)));
      kk[2] = std::min(1.0 - kk[0], std::max(0.0, 4.0 * std::get<2>(pairs[c]) - 1.0 + kk[0]));
      kk[1] = 1.0 - kk[0] - kk[2];
      for (double& x : kk) x = 0.98 * x + 0.02 / 3.0;
      ml_ibd(lik, M, kk);
      out[c].k0 = kk[0];
      out[c].k1 = kk[1];
      out[c].k2 = kk[2];
      out[c].kin = 0.25 * kk[1] + 0.5 * kk[2];
    }
  }
  cao.print(tick.date(), "robust PCA: final kinship and IBD sharing (k0, k1, k2) of", np,
            "related pairs from the final PCs: pass", t_pass, "s, evalAdmix", t_ea, "s, IBD", tick.reltime(), "s");
  return out;
}

static void write_outputs(Data* data, const Param& params, Eigen::Index N, const Mat2D& U, const Mat1D& w,
                          const std::vector<Pair>& pairs, const std::function<double(int, int)>& king_of,
                          const std::vector<FinalRel>* fin_in = nullptr) {
  write_matrix(params.fileout + ".eigvecs", U);
  write_matrix(params.fileout + ".eigvals", w);
  const std::vector<FinalRel> fin = fin_in ? *fin_in : final_relatedness(data, params, U, pairs);
  String1D ids = read_sample_ids(params);
  std::ofstream rp(params.fileout + ".relpairs");
  // KING: KING-robust; KIN_DETECT: kinship of the detection step (decides the
  // pair and its weight); KINSHIP, K0..K2: final estimates given the final PCs;
  // KIN_EVALADMIX: the --evaladmix kinship of the pair
  rp << "ID1\tID2\tKING\tKIN_DETECT\tKINSHIP\tK0\tK1\tK2\tKIN_EVALADMIX\n" << std::fixed << std::setprecision(6);
  for (size_t c = 0; c < pairs.size(); ++c) {
    const auto& [i, j, phi] = pairs[c];
    const std::string a = ids.size() == (size_t)N ? ids[i] : std::to_string(i);
    const std::string b = ids.size() == (size_t)N ? ids[j] : std::to_string(j);
    rp << a << "\t" << b << "\t" << king_of(i, j) << "\t" << phi << "\t" << fin[c].kin << "\t" << fin[c].k0 << "\t"
       << fin[c].k1 << "\t" << fin[c].k2 << "\t" << fin[c].kin_ea << "\n";
  }
  cao.print(tick.date(), "robust PCA: PCs saved to", params.fileout + ".eigvecs", "and", pairs.size(),
            "related pairs to", params.fileout + ".relpairs");
}

static void run_robust_operator(Data* data, const Param& params) {
  const Eigen::Index N = data->nsamples;
  const int k = params.k;
  const double tau = params.kin_min;
  const std::string& mode = params.robust;
  if (mode == "aarobust-kin" || mode == "diag-impute")
    cao.error("--robust " + mode + " is only available with --robust-engine dense");
  // candidate pairs: from --kinship, or KING-robust over all pairs computed
  // here from bit-packed genotypes stored during the first pass
  const bool internal_king = params.filekin.empty();
  std::vector<Pair> filepairs;
  std::vector<std::pair<int, int>> cand;
  std::map<std::pair<int, int>, double> king_map;
  if (!internal_king) {
    filepairs = read_kinship_pairs(params, N, std::min(params.king_screen, tau));
    for (const auto& [i, j, phi] : filepairs) {
      king_map[{i, j}] = phi;
      if (phi > params.king_screen) cand.emplace_back(i, j);
    }
    cao.print(tick.date(), "robust PCA (operator engine):", cand.size(), "candidate pairs from --kinship with kinship >",
              params.king_screen);
  }
  Eigen::Index nc = cand.size();
  PackedGeno pk;
  if (internal_king) pk.init(N, data->nsnps);
  // very large N: candidates from a sketch nearest-neighbour search instead of
  // KING over all pairs (--king-search)
  const bool use_sketch =
      internal_king && (params.king_search == "sketch" || (params.king_search == "auto" && N > params.king_sketch_min));
  RelSketch sk;
  if (use_sketch) sk.init(N, params.king_sketch_dim);

  // one pass: D, the diagonal of the raw Gram and its entries for the candidates
  Mat1D Dsum = Mat1D::Zero(N), Adiag = Mat1D::Zero(N), Ac = Mat1D::Zero(nc);
  uint64 Msites = 0;
  double edge_sum = 0;
  // the SNPs of a block are split over threads, each with its own sums
  auto add_block = [&](const GenoBlock& X) {
    const Eigen::Index b = X.cols();
    const int nt = std::max(1, std::min<int>(params.threads, (int)(b / 64)));
    std::vector<Mat1D> Ds(nt, Mat1D::Zero(N)), Ad(nt, Mat1D::Zero(N)), Acs(nt, Mat1D::Zero(nc));
    std::vector<double> es(nt, 0.0);
    const Eigen::Index step = (b + nt - 1) / nt;
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; ++t) {
      const Eigen::Index s0 = t * step, len = std::min(step, b - s0);
      if (len <= 0) continue;
      const Mat2D g = 2.0 * X.middleCols(s0, len);
      es[t] = site_noise_var(X.middleCols(s0, len));
      Ds[t] = (g.array() * (2.0 - g.array())).rowwise().sum().matrix();
      Ad[t] = g.array().square().rowwise().sum().matrix();
      const Mat2D gt = g.transpose();  // contiguous rows for the candidate dot products
      for (Eigen::Index c = 0; c < nc; ++c) Acs[t](c) = gt.col(cand[c].first).dot(gt.col(cand[c].second));
    }
    for (int t = 0; t < nt; ++t) {
      Dsum += Ds[t];
      Adiag += Ad[t];
      Ac += Acs[t];
      edge_sum += es[t];
    }
    if (internal_king) pk.add(X, Msites, params.threads);
    if (use_sketch) sk.add(X, Msites, params.threads);
    Msites += b;
  };
  tick.clock();
  if (!params.out_of_core) {
    if ((data->G.array() < 0).any()) cao.error("--robust requires complete genotypes (no missing values)");
    for (Eigen::Index s0 = 0; s0 < data->G.cols(); s0 += 4096)
      add_block(data->G.middleCols(s0, std::min<Eigen::Index>(4096, data->G.cols() - s0)));
  } else {
    data->F = Mat1D::Zero(data->nsnps);
    data->centered_geno_lookup = Arr2D::Zero(4, data->nsnps);
    data->check_file_offset_first_var();
    for (uint bi = 0; bi < data->nblocks; ++bi) {
      data->read_block_initial(data->start[bi], data->stop[bi], false);
      Mat2D X = data->G;
      for (Eigen::Index j = 0; j < X.cols(); ++j) X.col(j).array() += data->F(data->start[bi] + j);
      add_block(X);
    }
  }
  const double M = (double)Msites;
  if (internal_king) {
    cao.print(tick.date(), "robust PCA: first pass over", Msites, "sites in", tick.reltime(), "seconds");
    tick.clock();
    std::vector<Pair> kp;
    if (use_sketch) {
      sk.finish();
      uint64 nchecked = 0, nextended = 0;
      kp = king_sketch_pairs(pk, sk, params.king_neighbours, params.king_screen, params.threads, nchecked, nextended);
      sk = RelSketch();
      cao.print(tick.date(), "robust PCA (operator engine): sketch search ( dim", params.king_sketch_dim, ",",
                params.king_neighbours, "neighbours ) checked", nchecked, "pairs with KING-robust, extended the search for",
                nextended, "individuals");
    } else {
      kp = king_all_pairs(pk, params.king_screen, params.threads);
    }
    for (const auto& [i, j, kin] : kp) {
      king_map[{i, j}] = kin;
      cand.emplace_back(i, j);
    }
    nc = cand.size();
    filepairs.clear();
    for (const auto& [ij, kin] : king_map) filepairs.emplace_back(ij.first, ij.second, kin);
    Ac.resize(nc);
#pragma omp parallel for num_threads(params.threads)
    for (Eigen::Index c = 0; c < nc; ++c) Ac(c) = pk.gram(cand[c].first, cand[c].second);
    pk = PackedGeno();
    cao.print(tick.date(), "robust PCA (operator engine): KING-robust", use_sketch ? "on the sketch candidates" : "over all pairs",
              "in", tick.reltime(), "seconds,", nc, "candidate pairs with kinship >", params.king_screen);
  }
  Mat1D D = Dsum / M;
  Mat1D Acm = Ac / M;
  if (!internal_king)
    cao.print(tick.date(), "robust PCA: first pass over", Msites, "sites in", tick.reltime(), "seconds");

  GenoProduct gp{data, params, M};
  const Eigen::Index b = std::min<Eigen::Index>(N, k + 1 + 5);  // block size with oversampling
  auto rand_basis = [&]() {
    std::mt19937 rng(params.seed);
    std::normal_distribution<double> nd;
    Mat2D R(N, b);
    for (Eigen::Index c = 0; c < b; ++c)
      for (Eigen::Index r = 0; r < N; ++r) R(r, c) = nd(rng);
    Eigen::HouseholderQR<Mat2D> qr(R);
    return Mat2D(qr.householderQ() * Mat2D::Identity(N, b));
  };
  // final PCs of the whitened genotypes by PCAone's window-based RSVD (--svd 2):
  // left singular vectors of Sigma^-1/2 G (raw genotypes; Sigma^-1/2 applied
  // to each block as it is read, or once in place in-core), k + 1 of them with
  // the first (mean) dropped, mapped back by Sigma^1/2. In-core, G is restored
  // afterwards (Wh Wm = I, both from the same floored eigenvalues).
  auto winsvd_pcs = [&](const BlockDiag& Wm, const BlockDiag& Wh, Mat2D& Uo, Mat1D& wo) {
    const auto t0 = std::chrono::steady_clock::now();
    data->row_transform = [&Wm](Mat2D& G) { Wm.apply_inplace(G); };
    FancyRsvdOpData rsvd(data, k + 1, params.oversamples);
    rsvd.setFlags(false, false);
    rsvd.computeUSV(params.maxp, params.tol);
    data->row_transform = nullptr;
    if (!params.out_of_core) Wh.apply_inplace(data->G);
    Uo = Wh.apply(rsvd.U.middleCols(1, k));
    wo = (4.0 / M) * rsvd.S.segment(1, k).array().square().matrix();
    cao.print(tick.date(), "robust PCA: final PCs of the whitened genotypes by winSVD in",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), "seconds");
  };
  // eigenvectors of a fixed operator by subspace iteration to convergence
  // convergence tolerance (--robust-tol): relative residual of the first nstrict
  // (structure) Ritz pairs and change of the fit; the remaining output pairs
  // (in the flat noise spectrum, slow to converge) to max(tol, 1e-6)
  const double conv_tol = params.robust_tol;
  auto converge = [&](auto op, Mat2D V, Mat1D& theta, Mat2D& Vr, int nstrict) {
    // once the structure pairs have converged, the noise pairs get at most 10
    // more passes (about the accuracy of a randomized SVD; they are noise)
    double resid = 1;
    Mat1D res;
    bool ok = false;
    int extra = -1;
    for (int it = 0; it < 5000 && !ok; ++it) {
      ritz_step(op, V, theta, Vr, resid, nstrict, &res);
      if (resid < conv_tol && extra < 0) extra = 0;
      ok = resid < conv_tol && (res.head(k + 1).maxCoeff() < std::max(conv_tol, 1e-6) || extra++ >= 10);
    }
    if (!ok) cao.warn("--robust operator engine: subspace iteration residual ", resid);
  };
  Mat1D w, theta;
  Mat2D V, Vr, U;
  std::vector<Pair> pairs;
  if (mode == "detect-white" || mode == "frkin") {
    // fixed rank + kinship on the raw Gram with the diagonal free. Each
    // iteration is one pass: a subspace step on (A/M - S), then L and S are
    // updated from the Ritz pairs.
    Mat1D soff = Mat1D::Zero(nc), sdiag = Mat1D::Zero(N), Ldiag, Lc(nc);
    auto op = [&](const Mat2D& X) {
      Mat2D Y = gp.apply(X);
      Y -= sdiag.asDiagonal() * X;
      for (Eigen::Index c = 0; c < nc; ++c)
        if (soff(c) != 0) {
          const int i = cand[c].first, j = cand[c].second;
          Y.row(i) -= soff(c) * X.row(j);
          Y.row(j) -= soff(c) * X.row(i);
        }
      return Y;
    };
    V = rand_basis();
    int total = 0, rank = k + 1;
    uint64 pass_fit = 0;
    // detect-white: rank from the noise edge, at most k + 1 (as the dense engine)
    const double edge = mode == "detect-white" && !params.robust_fixed_rank ? noise_edge(edge_sum, M, N) : 0.0;
    for (int r = 1; r <= k + 1; ++r) {
      Mat1D Ldiag_old, Lc_old;
      double resid = 1;
      for (int it = 0; it < 5000; ++it) {
        ritz_step(op, V, theta, Vr, resid, r);
        Mat1D wr = theta.head(r);
        Mat2D Ur = Vr.leftCols(r);
        Ldiag = Ur.array().square().matrix() * wr;
        for (Eigen::Index c = 0; c < nc; ++c)
          Lc(c) = (Ur.row(cand[c].first).array() * Ur.row(cand[c].second).array() * wr.transpose().array()).sum();
        for (Eigen::Index c = 0; c < nc; ++c) {
          const int i = cand[c].first, j = cand[c].second;
          const double R = Acm(c) - Lc(c);
          soff(c) = R / (2.0 * std::sqrt(std::max(D(i) * D(j), 1e-24))) > tau ? R : 0.0;
        }
        sdiag = Adiag / M - Ldiag;  // diagonal free
        ++total;
        if (it > 0 && resid < conv_tol) {
          double ch = (Ldiag - Ldiag_old).cwiseAbs().maxCoeff();
          if (nc > 0) ch = std::max(ch, (Lc - Lc_old).cwiseAbs().maxCoeff());
          if (ch < conv_tol) break;
        }
        Ldiag_old = Ldiag;
        Lc_old = Lc;
      }
      if (edge > 0 && r < k + 1) {
        // the next eigenvalue of A/M - S at the converged fit
        if (!next_above_edge(op, V, r, edge)) {
          rank = r;
          break;
        }
      }
    }
    for (Eigen::Index c = 0; c < nc; ++c)
      if (soff(c) != 0)
        pairs.emplace_back(cand[c].first, cand[c].second,
                           (Acm(c) - Lc(c)) /
                               (2.0 * std::sqrt(std::max(D(cand[c].first) * D(cand[c].second), 1e-24))));
    pass_fit = gp.passes;
    cao.print(tick.date(), "robust PCA: fixed rank + kinship fit, rank", rank, "(at most", k + 1, ", noise edge", edge, "),", total, "iterations,", pass_fit, "passes,", pairs.size(),
              "related pairs detected");
    if (mode == "frkin") {
      U = Vr.block(0, 1, N, k);
      w = theta.segment(1, k);
    } else {
      Mat1D noise = Adiag / M - Ldiag;
      BlockDiag Wm, Wh;
      build_whitener(noise, pairs, Wm, Wh);
      if (params.robust_pcs == "winsvd") {
        winsvd_pcs(Wm, Wh, U, w);
      } else {
        auto wop = [&](const Mat2D& X) { return Wm.apply(gp.apply(Wm.apply(X))); };
        // warm start: the structure part of Sigma^-1/2 A Sigma^-1/2 spans
        // Sigma^-1/2 times the fitted eigenvectors
        Eigen::HouseholderQR<Mat2D> qr0(Wm.apply(V));
        converge(wop, Mat2D(qr0.householderQ() * Mat2D::Identity(N, V.cols())), theta, Vr, rank);
        U = Wh.apply(Vr.block(0, 1, N, k));
        w = theta.segment(1, k);
      }
    }
  } else if (mode == "cswhite") {
    cao.warn("--robust cswhite corrects the diagonal with heterozygosity, which assumes HWE within individuals");
    for (const auto& [i, j, phi] : filepairs)
      if (phi >= tau) pairs.emplace_back(i, j, phi);
    BlockDiag Wm, Wh;
    build_whitener(D, pairs, Wm, Wh);
    if (params.robust_pcs == "winsvd") {
      winsvd_pcs(Wm, Wh, U, w);
    } else {
      auto wop = [&](const Mat2D& X) { return Wm.apply(gp.apply(Wm.apply(X))); };
      converge(wop, rand_basis(), theta, Vr, k + 1);
      U = Wh.apply(Vr.block(0, 1, N, k));
      w = theta.segment(1, k);
    }
  }
  cao.print(tick.date(), "robust PCA:", gp.passes, "passes over the genotypes in total,", gp.secs,
            "seconds in the genotype products");
  write_outputs(data, params, N, U, w, pairs, [&](int i, int j) {
    auto it = king_map.find({std::min(i, j), std::max(i, j)});
    return it == king_map.end() ? NAN : it->second;
  });
}

// ---- dwg: detect-white on the GRM scale (scripts/relatepca/dwg_loc.py) -------
// The matrix is the SNP-standardised but uncentred Gram A_s (centring would
// spread each individual's heterozygosity over its off-diagonal entries).
// Kinship is scaled by the fitted noise v = A_s,ii - L_ii (no HWE within
// individuals). The rank comes from the noise edge, but an eigenvector resting
// on fewer than 4 individuals with a related pair among them (kinship >
// 2^-4.5) is a family, not structure: the pairs among its top individuals
// become candidates and the rank is not raised. Candidates: KING-robust >
// screen, then evalAdmix kinship from the structure axes (admixture-aware),
// confirmed by a k0 moment check (IBD: observed / expected opposite
// homozygotes < 0.8), iterated until the pairs are stable.
static const double DWG_TAU3 = 0.04419417382415922;  // 2^-4.5
static const double DWG_NEFF = 4.0;
static const double DWG_K0_MAX = 0.8;  // 2nd degree about 0.5 (0.7 cross-ancestry with low realised kinship)

// u: an eigenvector; R: residual without this axis; v: noise
static bool dwg_family_axis(const Mat1D& u0, const Mat2D& R, const Mat1D& v, std::vector<int>& top) {
  const Mat1D u = u0 / u0.norm();
  const double neff = 1.0 / u.array().pow(4).sum();
  std::vector<int> idx(u.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return std::abs(u(a)) > std::abs(u(b)); });
  top.assign(idx.begin(), idx.begin() + std::max<int>(2, (int)std::ceil(neff)));
  if (neff >= DWG_NEFF) return false;
  double mx = -1e300;
  for (size_t a = 0; a < top.size(); ++a)
    for (size_t b = a + 1; b < top.size(); ++b) {
      const int i = top[a], j = top[b];
      mx = std::max(mx, R(i, j) / (2.0 * std::sqrt(std::max(v(i), 1e-6) * std::max(v(j), 1e-6))));
    }
  return mx > DWG_TAU3;
}

struct DwgFit {
  Mat2D L, S, V;  // V: the rank-r eigenvectors of the fit (first = mean component)
  Mat1D w;
  int r = 1;
  std::vector<Pair> pairs;
  Mat2D axes;  // structure axes (family axes dropped), for the evalAdmix screen
};

// rho (optional): per-individual factor on the noise in the kinship rule
// (genotype reliability with genotype likelihoods; 1 for genotypes)
static DwgFit dwg_fit(const Mat2D& A, double tau, MatB cand, double edge, int rmax, int maxit = 500,
                      const Mat1D& rho_in = Mat1D()) {
  const Eigen::Index n = A.rows();
  const Mat1D rho = rho_in.size() == n ? rho_in : Mat1D::Ones(n);
  DwgFit F;
  F.S = Mat2D::Zero(n, n);
  Mat2D Vb = rand_orth(n, std::min<Eigen::Index>(n, rmax + 7), 3);
  int r = 1;
  while (true) {
    Mat2D Lold;
    for (int it = 0; it < maxit; ++it) {
      top_eig_warm(A - F.S, r, Vb, F.w, F.V);
      F.L.noalias() = F.V * F.w.asDiagonal() * F.V.transpose();
      const Mat2D R = A - F.L;
      const Mat1D v = R.diagonal().cwiseMax(1e-6).cwiseProduct(rho);
      const MatB sel = kin_select(R, v, tau, cand);
      F.S = sel.select(R, Mat2D::Zero(n, n));
      F.S.diagonal() = R.diagonal();
      if (it > 0 && (F.L - Lold).cwiseAbs().maxCoeff() < 1e-10) break;
      Lold = F.L;
    }
    if (r >= std::min<Eigen::Index>(rmax, n - 1)) break;
    Mat1D w2;
    Mat2D V2;
    top_eig_warm(A - F.S, r + 1, Vb, w2, V2);
    if (w2(r) <= edge) break;
    const Mat2D R = A - F.L;
    std::vector<int> top;
    if (dwg_family_axis(V2.col(r), R, R.diagonal().cwiseMax(1e-6).cwiseProduct(rho), top)) {
      bool added = false;
      for (size_t a = 0; a < top.size(); ++a)
        for (size_t b = a + 1; b < top.size(); ++b)
          if (!cand(top[a], top[b])) cand(top[a], top[b]) = cand(top[b], top[a]) = added = true;
      if (!added) break;  // a family axis with nothing to add (e.g. cousins): stop
      continue;
    }
    ++r;
  }
  F.r = r;
  const Mat2D R = A - F.L;
  const Mat1D v = R.diagonal().cwiseMax(1e-6);
  for (Eigen::Index j = 0; j < n; ++j)
    for (Eigen::Index i = 0; i < j; ++i)
      if (std::abs(F.S(i, j)) > 1e-12) F.pairs.emplace_back(i, j, R(i, j) / (2.0 * std::sqrt(v(i) * v(j))));
  // structure axes: eigenvectors 2..r, family axes dropped (residual with the axis put back)
  std::vector<int> keep;
  for (int j = 1; j < r; ++j) {
    const Mat2D Rj = R + F.w(j) * F.V.col(j) * F.V.col(j).transpose();
    std::vector<int> top;
    if (!dwg_family_axis(F.V.col(j), Rj, Rj.diagonal().cwiseMax(1e-6).cwiseProduct(rho), top)) keep.push_back(j);
  }
  if (keep.empty() && r > 1) keep.push_back(1);
  F.axes = Mat2D(n, keep.size());
  for (size_t c = 0; c < keep.size(); ++c) F.axes.col(c) = F.V.col(keep[c]);
  return F;
}

// evalAdmix kinship (b - c)/2 for all pairs from the raw Gram A = G G'
// (0/1/2), the per-individual mean genotype and heterozygosity, given axes
static Mat2D dwg_evaladmix(const Mat2D& A, const Mat1D& gbar, const Mat1D& d, double M, const Mat2D& axes) {
  const Eigen::Index n = A.rows();
  Mat2D V(n, axes.cols() + 1);
  V.leftCols(axes.cols()) = axes;
  V.col(axes.cols()).setOnes();
  const Mat2D P = V * (V.transpose() * V).completeOrthogonalDecomposition().pseudoInverse() * V.transpose();
  const Mat2D IP = Mat2D::Identity(n, n) - P;
  auto cov2cor = [](const Mat2D& C) {
    const Mat1D s = C.diagonal().cwiseMax(1e-300).cwiseSqrt();
    return Mat2D(s.cwiseInverse().asDiagonal() * C * s.cwiseInverse().asDiagonal());
  };
  const Mat2D b = cov2cor(IP * (A - M * gbar * gbar.transpose()) * IP);
  const Mat2D c = cov2cor(IP * d.asDiagonal() * IP);
  Mat2D kin = (b - c) / 2.0;
  kin.diagonal().setZero();
  return kin;
}

// k0 moment for pairs: observed (I0 counts) / expected opposite homozygotes,
// individual allele frequencies from [axes, 1] with the pair left out
static std::vector<double> dwg_k0(Data* data, const Param& params, const Mat2D& axes, const Mat2D& I0,
                                  const std::vector<std::pair<int, int>>& pairs) {
  const Eigen::Index N = data->nsamples, m1 = axes.cols() + 1, M = data->nsnps;
  std::vector<double> out(pairs.size(), 1.0);
  if (pairs.empty()) return out;
  Mat2D V(N, m1);
  V.leftCols(m1 - 1) = axes;
  V.col(m1 - 1).setOnes();
  std::vector<int> sub(N, -1), members;
  for (const auto& [i, j] : pairs)
    for (int x : {i, j})
      if (sub[x] < 0) {
        sub[x] = members.size();
        members.push_back(x);
      }
  const Eigen::Index ni = members.size();
  Mat2D R(m1, M);
  std::vector<int8_t> Gsub(ni * M);
  Eigen::Index s0 = 0;
  auto add_block = [&](const GenoBlock& X) {  // 0..1 scale
    const Eigen::Index b = X.cols();
    R.middleCols(s0, b).noalias() = 2.0 * (V.transpose() * X);
    for (Eigen::Index t = 0; t < ni; ++t)
      for (Eigen::Index j = 0; j < b; ++j) Gsub[t * M + s0 + j] = (int8_t)(2.0 * X(members[t], j) + 0.5);
    s0 += b;
  };
  if (!params.out_of_core) {
    for (Eigen::Index c = 0; c < data->G.cols(); c += 4096)
      add_block(data->G.middleCols(c, std::min<Eigen::Index>(4096, data->G.cols() - c)));
  } else {
    data->check_file_offset_first_var();
    for (uint bi = 0; bi < data->nblocks; ++bi) {
      data->read_block_initial(data->start[bi], data->stop[bi], false);
      add_block(data->G);
    }
  }
  const Mat2D VtV = V.transpose() * V;
#pragma omp parallel for num_threads(params.threads)
  for (size_t c = 0; c < pairs.size(); ++c) {
    const int i = pairs[c].first, j = pairs[c].second;
    const Mat1D vi = V.row(i).transpose(), vj = V.row(j).transpose();
    const Mat2D Ai = (VtV - vi * vi.transpose() - vj * vj.transpose()).completeOrthogonalDecomposition().pseudoInverse();
    const Mat1D wi = Ai * vi, wj = Ai * vj;
    const Mat1D ri = R.transpose() * wi, rj = R.transpose() * wj;
    const double aa = vi.dot(wi), ab = vi.dot(wj), bb = vj.dot(wj);
    const int8_t *gi = &Gsub[sub[i] * M], *gj = &Gsub[sub[j] * M];
    double ex = 0;
    for (Eigen::Index t = 0; t < M; ++t) {
      double pi = 0.5 * (ri(t) - aa * gi[t] - ab * gj[t]), pj = 0.5 * (rj(t) - ab * gi[t] - bb * gj[t]);
      pi = std::min(std::max(pi, 1e-3), 1.0 - 1e-3);
      pj = std::min(std::max(pj, 1e-3), 1.0 - 1e-3);
      ex += pi * pi * (1 - pj) * (1 - pj) + (1 - pi) * (1 - pi) * pj * pj;
    }
    double obs = 0;
    if (I0.size() > 0) {
      obs = I0(i, j);
    } else {
      for (Eigen::Index t = 0; t < M; ++t) obs += (gi[t] + gj[t] == 2) && (gi[t] != 1);
    }
    out[c] = obs / std::max(ex, 1e-12);
  }
  return out;
}


// ---- dwg in the operator engine (no N x N matrix) ----------------------------
// Z += G_s (G_s' V) with G_s = G diag(scale) for one block, SNPs split over
// threads (scale per site: 2/sqrt(2f(1-f)) on the 0..1 scale, 0 if monomorphic)
static void add_GGtV_w(const GenoBlock& G, const Mat1D& scale, const Mat2D& V, Mat2D& Z, int nthreads) {
  const Eigen::Index M = G.cols();
  const int nt = std::max(1, std::min<int>(nthreads, (int)(M / 256)));
  const Eigen::Index nchunk = 4 * nt, step = (M + nchunk - 1) / nchunk;
  std::vector<Mat2D> Zt(nt, Mat2D::Zero(Z.rows(), Z.cols()));
#pragma omp parallel for num_threads(nt) schedule(dynamic)
  for (Eigen::Index c = 0; c < nchunk; ++c) {
    const Eigen::Index s0 = c * step, len = std::min(step, M - s0);
    if (len <= 0) continue;
    // G_s G_s' V = G (diag(scale^2) G' V): scale the small product, not the genotypes
    Mat2D T = G.middleCols(s0, len).transpose() * V;
    T = scale.segment(s0, len).array().square().matrix().asDiagonal() * T;
    Zt[omp_get_thread_num()].noalias() += G.middleCols(s0, len) * T;
  }
  for (const auto& z : Zt) Z += z;
}

// one pass over the genotypes (raw, 0..1 scale) calling fn(block, first site)
static void for_each_block(Data* data, const Param& params, const std::function<void(const GenoBlock&, Eigen::Index)>& fn) {
  if (!params.out_of_core) {
    for (Eigen::Index c = 0; c < data->G.cols(); c += 4096)
      fn(data->G.middleCols(c, std::min<Eigen::Index>(4096, data->G.cols() - c)), c);
  } else {
    data->check_file_offset_first_var();
    for (uint bi = 0; bi < data->nblocks; ++bi) {
      data->read_block_initial(data->start[bi], data->stop[bi], false);
      fn(data->G, data->start[bi]);
    }
  }
}

// raw and SNP-weighted Gram entries (sum over sites of g_i g_j, and of
// g_i g_j / (2f(1-f))) for a list of pairs, in one pass
static void pair_sums(Data* data, const Param& params, const Mat1D& wscale2, const std::vector<std::pair<int, int>>& pairs,
                      Mat1D& raw, Mat1D& wtd) {
  const Eigen::Index N = data->nsamples, np = pairs.size();
  raw = Mat1D::Zero(np);
  wtd = Mat1D::Zero(np);
  if (np == 0) return;
  std::vector<int> sub(N, -1), members;
  for (const auto& [i, j] : pairs)
    for (int x : {i, j})
      if (sub[x] < 0) {
        sub[x] = members.size();
        members.push_back(x);
      }
  const Eigen::Index ni = members.size();
  const int nt = std::max(1, (int)params.threads);
  for_each_block(data, params, [&](const GenoBlock& X, Eigen::Index first) {
    const Eigen::Index b = X.cols(), cs = 256, nch = (b + cs - 1) / cs;
    std::vector<Mat1D> rt(nt, Mat1D::Zero(np)), wt(nt, Mat1D::Zero(np));
#pragma omp parallel for num_threads(nt) schedule(dynamic)
    for (Eigen::Index ch = 0; ch < nch; ++ch) {
      const Eigen::Index c0 = ch * cs, len = std::min(cs, b - c0);
      Mat2D xt(len, ni);
      for (Eigen::Index j = 0; j < len; ++j)
        for (Eigen::Index t = 0; t < ni; ++t) xt(j, t) = 2.0 * X(members[t], c0 + j);
      const Mat1D ws = wscale2.segment(first + c0, len);
      const int th = omp_get_thread_num();
      for (Eigen::Index c = 0; c < np; ++c) {
        const auto a = xt.col(sub[pairs[c].first]), bb = xt.col(sub[pairs[c].second]);
        rt[th](c) += a.dot(bb);
        wt[th](c) += (a.array() * bb.array() * ws.array()).sum();
      }
    }
    for (int t = 0; t < nt; ++t) {
      raw += rt[t];
      wtd += wt[t];
    }
  });
}

static void run_dwg_operator(Data* data, const Param& params) {
  const Eigen::Index N = data->nsamples, Msnp = data->nsnps;
  const int k = params.k;
  const double tau = params.kin_min;
  cao.print(tick.date(), "robust PCA ( dwg, operator engine ): N =", N, ", k =", k, ", tau =", tau,
            ", KING screen =", params.king_screen);
  const bool internal_king = params.filekin.empty();
  std::vector<std::pair<int, int>> cand;
  std::map<std::pair<int, int>, double> king_map;
  if (!internal_king)
    for (const auto& [i, j, phi] : read_kinship_pairs(params, N, params.king_screen))
      if (phi > params.king_screen) {
        king_map[{i, j}] = phi;
        cand.emplace_back(i, j);
      }
  PackedGeno pk;
  if (internal_king) pk.init(N, Msnp);
  const bool use_sketch =
      internal_king && (params.king_search == "sketch" || (params.king_search == "auto" && N > params.king_sketch_min));
  RelSketch sk;  // always: also the residual sketch of the evalAdmix screen
  sk.init(N, params.king_sketch_dim);

  // ---- pass 1 ----------------------------------------------------------------
  Mat1D Dsum = Mat1D::Zero(N), Adiag = Mat1D::Zero(N), Asdiag = Mat1D::Zero(N), gsum = Mat1D::Zero(N);
  Mat1D wscale = Mat1D::Zero(Msnp), wscale2 = Mat1D::Zero(Msnp);  // 2/sqrt(w) (0..1 scale), 1/w (g scale)
  double edge_s = 0;
  uint64 Msites = 0, Ms = 0;
  auto add_block = [&](const GenoBlock& X) {
    const Eigen::Index b = X.cols();
    const int nt = std::max(1, std::min<int>(params.threads, (int)(b / 64)));
    std::vector<Mat1D> Ds(nt, Mat1D::Zero(N)), Ad(nt, Mat1D::Zero(N)), As(nt, Mat1D::Zero(N)), gs(nt, Mat1D::Zero(N));
    std::vector<double> es(nt, 0.0);
    std::vector<uint64> ms(nt, 0);
    const Eigen::Index step = (b + nt - 1) / nt;
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; ++t) {
      const Eigen::Index s0 = t * step, len = std::min(step, b - s0);
      if (len <= 0) continue;
      const Mat2D g = 2.0 * X.middleCols(s0, len);
      Ds[t] = (g.array() * (2.0 - g.array())).rowwise().sum().matrix();
      Ad[t] = g.array().square().rowwise().sum().matrix();
      gs[t] = g.rowwise().sum();
      for (Eigen::Index j = 0; j < len; ++j) {
        const double fj = g.col(j).mean() / 2.0, w = 2.0 * fj * (1.0 - fj);
        if (w > VAR_TOL) {
          wscale(Msites + s0 + j) = 2.0 / std::sqrt(w);
          wscale2(Msites + s0 + j) = 1.0 / w;
          As[t] += g.col(j).array().square().matrix() / w;
          const double eg2 = 2.0 * fj * (1.0 - fj) + 4.0 * fj * fj;
          es[t] += (eg2 * eg2 - 16.0 * fj * fj * fj * fj) / (w * w);
          ++ms[t];
        }
      }
    }
    for (int t = 0; t < nt; ++t) {
      Dsum += Ds[t];
      Adiag += Ad[t];
      Asdiag += As[t];
      gsum += gs[t];
      edge_s += es[t];
      Ms += ms[t];
    }
    if (internal_king) pk.add(X, Msites, params.threads);
    sk.add(X, Msites, params.threads);
    Msites += b;
  };
  tick.clock();
  if (!params.out_of_core) {
    if ((data->G.array() < 0).any()) cao.error("--robust requires complete genotypes (no missing values)");
    for (Eigen::Index s0 = 0; s0 < data->G.cols(); s0 += 4096)
      add_block(data->G.middleCols(s0, std::min<Eigen::Index>(4096, data->G.cols() - s0)));
  } else {
    data->F = Mat1D::Zero(Msnp);
    data->centered_geno_lookup = Arr2D::Zero(4, Msnp);
    data->check_file_offset_first_var();
    for (uint bi = 0; bi < data->nblocks; ++bi) {
      data->read_block_initial(data->start[bi], data->stop[bi], false);
      Mat2D X = data->G;
      for (Eigen::Index j = 0; j < X.cols(); ++j) X.col(j).array() += data->F(data->start[bi] + j);
      add_block(X);
    }
  }
  const double M = (double)Msites, Msd = (double)Ms;
  const Mat1D D = Dsum / M, gbar = gsum / M;
  Asdiag /= Msd;
  const double edge = 2.0 * std::sqrt(edge_s / (Msd * Msd)) * std::sqrt((double)N);
  cao.print(tick.date(), "robust PCA: first pass over", Msites, "sites in", tick.reltime(), "seconds");
  Eigen::MatrixXf Yraw = sk.Y;  // N x dim, for the residual sketch
  if (internal_king) {
    tick.clock();
    std::vector<Pair> kp;
    if (use_sketch) {
      sk.finish();
      uint64 nchecked = 0, nextended = 0;
      kp = king_sketch_pairs(pk, sk, params.king_neighbours, params.king_screen, params.threads, nchecked, nextended);
    } else {
      kp = king_all_pairs(pk, params.king_screen, params.threads);
    }
    for (const auto& [i, j, kin] : kp) {
      king_map[{i, j}] = kin;
      cand.emplace_back(i, j);
    }
    pk = PackedGeno();
    cao.print(tick.date(), "robust PCA (dwg): KING-robust", use_sketch ? "on the sketch candidates" : "over all pairs", "in",
              tick.reltime(), "seconds,", cand.size(), "candidate pairs");
  }
  sk = RelSketch();

  // GRM-scale operator A_s V and the raw Gram operator (1/M) G G' V (g scale)
  uint64 passes = 0;
  auto As_apply = [&](const Mat2D& V) {
    ++passes;
    Mat2D Z = Mat2D::Zero(V.rows(), V.cols());
    for_each_block(data, params, [&](const GenoBlock& X, Eigen::Index first) {
      const Mat1D sc = wscale.segment(first, X.cols());
      add_GGtV_w(X, sc, V, Z, params.threads);
    });
    return Mat2D(Z / Msd);
  };
  GenoProduct gp{data, params, M};
  // exact A_s entries of the candidate pairs (popcounts cannot weight SNPs)
  std::map<std::pair<int, int>, double> asmap;
  auto add_cands = [&](const std::vector<std::pair<int, int>>& cs) {
    std::vector<std::pair<int, int>> todo;
    for (const auto& c : cs)
      if (!asmap.count(c)) todo.push_back(c);
    Mat1D raw, wtd;
    pair_sums(data, params, wscale2, todo, raw, wtd);
    for (size_t c = 0; c < todo.size(); ++c) asmap[todo[c]] = wtd(c) / Msd;
  };
  tick.clock();
  add_cands(cand);

  const Eigen::Index b = std::min<Eigen::Index>(N, k + 1 + 5);
  Mat2D V = rand_orth(N, b, params.seed);
  const double conv_tol = params.robust_tol;

  struct OpFit {
    Mat2D Ur;
    Mat1D wr, Ldiag, v;
    int r = 1;
    std::vector<Pair> pairs;
    Mat2D axes;
  };
  // A_s entries among a few individuals: columns of A_s by one product pass
  auto as_block = [&](const std::vector<int>& idx) {
    Mat2D E = Mat2D::Zero(N, idx.size());
    for (size_t t = 0; t < idx.size(); ++t) E(idx[t], t) = 1.0;
    const Mat2D Z = As_apply(E);
    Mat2D out(idx.size(), idx.size());
    for (size_t a = 0; a < idx.size(); ++a)
      for (size_t c = 0; c < idx.size(); ++c) out(a, c) = Z(idx[a], c);
    return out;
  };
  auto top_of = [&](const Mat1D& u0, double& neff) {
    const Mat1D u = u0 / u0.norm();
    neff = 1.0 / u.array().pow(4).sum();
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    const int nt = std::max<int>(2, (int)std::ceil(neff));
    std::partial_sort(idx.begin(), idx.begin() + nt, idx.end(),
                      [&](int a, int c) { return std::abs(u(a)) > std::abs(u(c)); });
    return std::vector<int>(idx.begin(), idx.begin() + nt);
  };
  auto fit = [&](std::vector<std::pair<int, int>> cs) {
    OpFit F;
    std::vector<double> asc;
    for (const auto& c : cs) asc.push_back(asmap.at(c));
    Eigen::Index nc = cs.size();
    Mat1D soff = Mat1D::Zero(nc), sdiag = Mat1D::Zero(N), Lc;
    auto op = [&](const Mat2D& X) {
      Mat2D Y = As_apply(X);
      Y -= sdiag.asDiagonal() * X;
      for (Eigen::Index c = 0; c < nc; ++c)
        if (soff(c) != 0) {
          const int i = cs[c].first, j = cs[c].second;
          Y.row(i) -= soff(c) * X.row(j);
          Y.row(j) -= soff(c) * X.row(i);
        }
      return Y;
    };
    Mat1D theta;
    Mat2D Vr;
    int r = 1;
    std::set<std::pair<int, int>> inc(cs.begin(), cs.end());
    while (true) {
      Mat1D Ldiag_old, Lc_old;
      double resid = 1;
      // at most 500 passes per rank: near-degenerate structure eigenvalues (e.g.
      // a cousin axis next to a weak ancestry axis) converge very slowly
      int it = 0;
      for (; it < 500; ++it) {
        ritz_step(op, V, theta, Vr, resid, r);
        F.wr = theta.head(r);
        F.Ur = Vr.leftCols(r);
        F.Ldiag = F.Ur.array().square().matrix() * F.wr;
        F.v = (Asdiag - F.Ldiag).cwiseMax(1e-6);
        Lc.resize(nc);
        for (Eigen::Index c = 0; c < nc; ++c)
          Lc(c) = (F.Ur.row(cs[c].first).array() * F.Ur.row(cs[c].second).array() * F.wr.transpose().array()).sum();
        for (Eigen::Index c = 0; c < nc; ++c) {
          const int i = cs[c].first, j = cs[c].second;
          const double R = asc[c] - Lc(c);
          soff(c) = R / (2.0 * std::sqrt(F.v(i) * F.v(j))) > tau ? R : 0.0;
        }
        sdiag = Asdiag - F.Ldiag;
        if (it > 0 && resid < conv_tol) {
          double ch = (F.Ldiag - Ldiag_old).cwiseAbs().maxCoeff();
          if (nc > 0) ch = std::max(ch, (Lc - Lc_old).cwiseAbs().maxCoeff());
          if (ch < conv_tol) break;
        }
        Ldiag_old = F.Ldiag;
        Lc_old = Lc;
      }
      if (it == 500) cao.warn("--robust dwg (operator engine): the fit at rank ", r, " did not converge in 500 passes");
      if (r >= std::min<Eigen::Index>(k + 1, N - 1)) break;
      // the next eigenpair of A_s - S at the converged fit
      for (int it = 0; it < 300; ++it) {
        ritz_step(op, V, theta, Vr, resid, r + 1);
        if (resid < 1e-6) break;
      }
      if (theta(r) <= edge) break;
      double neff;
      const std::vector<int> top = top_of(Vr.col(r), neff);
      if (neff < DWG_NEFF) {
        const Mat2D Ab = as_block(top);
        double mx = -1e300;
        for (size_t a = 0; a < top.size(); ++a)
          for (size_t c = a + 1; c < top.size(); ++c) {
            const int i = top[a], j = top[c];
            const double Lij = (F.Ur.row(i).array() * F.Ur.row(j).array() * F.wr.transpose().array()).sum();
            mx = std::max(mx, (Ab(a, c) - Lij) / (2.0 * std::sqrt(F.v(i) * F.v(j))));
          }
        if (mx > DWG_TAU3) {
          bool added = false;
          for (size_t a = 0; a < top.size(); ++a)
            for (size_t c = a + 1; c < top.size(); ++c) {
              const auto pr = std::make_pair(std::min(top[a], top[c]), std::max(top[a], top[c]));
              if (!inc.count(pr)) {
                inc.insert(pr);
                cs.push_back(pr);
                asmap[pr] = Ab(a, c);
                asc.push_back(Ab(a, c));
                added = true;
              }
            }
          if (!added) break;
          nc = cs.size();
          soff.conservativeResize(nc);
          soff.tail(nc - Lc.size()).setZero();
          continue;
        }
      }
      ++r;
    }
    F.r = r;
    for (Eigen::Index c = 0; c < nc; ++c)
      if (soff(c) != 0) {
        const int i = cs[c].first, j = cs[c].second;
        F.pairs.emplace_back(i, j, (asc[c] - Lc(c)) / (2.0 * std::sqrt(F.v(i) * F.v(j))));
      }
    std::sort(F.pairs.begin(), F.pairs.end());
    // structure axes: 2..r without family axes
    std::vector<int> keep;
    for (int j = 1; j < r; ++j) {
      double neff;
      const std::vector<int> top = top_of(F.Ur.col(j), neff);
      bool fam = false;
      if (neff < DWG_NEFF) {
        const Mat2D Ab = as_block(top);
        double mx = -1e300;
        for (size_t a = 0; a < top.size(); ++a)
          for (size_t c = a + 1; c < top.size(); ++c) {
            const int i = top[a], q = top[c];
            const double uj = F.wr(j) * F.Ur(i, j) * F.Ur(q, j);
            const double Lij = (F.Ur.row(i).array() * F.Ur.row(q).array() * F.wr.transpose().array()).sum() - uj;
            const double vi = std::max(F.v(i) + F.wr(j) * F.Ur(i, j) * F.Ur(i, j), 1e-6);
            const double vq = std::max(F.v(q) + F.wr(j) * F.Ur(q, j) * F.Ur(q, j), 1e-6);
            mx = std::max(mx, (Ab(a, c) - Lij) / (2.0 * std::sqrt(vi * vq)));
          }
        fam = mx > DWG_TAU3;
      }
      if (!fam) keep.push_back(j);
    }
    if (keep.empty() && r > 1) keep.push_back(1);
    F.axes = Mat2D(N, keep.size());
    for (size_t c = 0; c < keep.size(); ++c) F.axes.col(c) = F.Ur.col(keep[c]);
    return F;
  };

  const auto tf0 = std::chrono::steady_clock::now();
  OpFit F = fit(cand);
  const double t_fit = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
  const uint64 passes_fit = passes;
  const auto ts0 = std::chrono::steady_clock::now();
  // key: pairs of the current fit; a refit that does not change them ends the
  // screen (no second neighbour search for candidates that did not become pairs)
  std::set<std::pair<int, int>> key, used(cand.begin(), cand.end());
  for (const auto& [i, j, phi] : F.pairs) key.emplace(i, j);
  int rounds = 0, n_ea = 0;
  for (; rounds < 5; ++rounds) {
    // evalAdmix candidates: nearest neighbours in the sketch with the structure
    // axes (and the mean) projected out, then the exact evalAdmix kinship
    Mat2D Vx(N, F.axes.cols() + 1);
    Vx.leftCols(F.axes.cols()) = F.axes;
    Vx.col(F.axes.cols()).setOnes();
    Eigen::HouseholderQR<Mat2D> qr(Vx);
    const Eigen::MatrixXf Q = Mat2D(qr.householderQ() * Mat2D::Identity(N, Vx.cols())).cast<float>();
    RelSketch rs;
    rs.N = N;
    rs.dim = Yraw.cols();
    rs.Y = Yraw - Q * (Q.transpose() * Yraw);
    rs.finish();
    std::vector<std::vector<int>> nbr(N);
    const int m = (int)std::min<Eigen::Index>(params.king_neighbours, N - 1);
#pragma omp parallel for num_threads(params.threads) schedule(dynamic)
    for (Eigen::Index r0 = 0; r0 < N; r0 += 64) top_neighbours(rs.Y, r0, std::min(N, r0 + 64), m, nbr);
    // pre-filter: the residual-sketch correlation estimates 2 x kinship (sd
    // about 1/sqrt(dim) for unrelated pairs, so the best unrelated neighbour
    // reaches about 0.1 at N = 20,000). Only pairs that can pass tau (0.088,
    // a correlation of about 0.18) matter; cross-ancestry 2nd-degree pairs had
    // evalAdmix kinship of about 0.085. Pairs below 0.12 (kinship 0.06) are
    // not worth the exact check.
    std::set<std::pair<int, int>> ce_set;
    for (Eigen::Index i = 0; i < N; ++i)
      for (int j : nbr[i]) {
        const auto pr = std::make_pair((int)std::min<Eigen::Index>(i, j), (int)std::max<Eigen::Index>(i, j));
        if (!used.count(pr) && rs.Y.col(i).dot(rs.Y.col(j)) > 0.12f) ce_set.insert(pr);
      }
    rs = RelSketch();
    std::vector<std::pair<int, int>> ce(ce_set.begin(), ce_set.end());
    // exact evalAdmix kinship for these pairs (projection estimator, as --evaladmix)
    Mat1D raw, wtd;
    pair_sums(data, params, wscale2, ce, raw, wtd);
    const Mat2D AV = M * gp.apply(Vx);  // G G' Vx on the g scale
    const Mat2D Pinv = (Vx.transpose() * Vx).completeOrthogonalDecomposition().pseudoInverse();
    const Mat2D CV = AV - M * gbar * (gbar.transpose() * Vx);
    const Mat2D VCV = Vx.transpose() * CV, VDV = Vx.transpose() * D.asDiagonal() * Vx, VP = Vx * Pinv;
    auto cw = [&](Eigen::Index i, Eigen::Index j, double Cij) {
      return Cij - VP.row(i).dot(CV.row(j)) - CV.row(i).dot(VP.row(j)) + VP.row(i) * VCV * VP.row(j).transpose();
    };
    auto ew = [&](Eigen::Index i, Eigen::Index j) {
      const double Pij = VP.row(i).dot(Vx.row(j));
      return (i == j ? D(i) : 0.0) - Pij * D(j) - D(i) * Pij + VP.row(i) * VDV * VP.row(j).transpose();
    };
    std::vector<std::pair<int, int>> pass_ea;
    for (size_t c = 0; c < ce.size(); ++c) {
      const int i = ce[c].first, j = ce[c].second;
      const double bij = cw(i, j, raw(c) - M * gbar(i) * gbar(j)) /
                         std::sqrt(std::max(cw(i, i, Adiag(i) - M * gbar(i) * gbar(i)) *
                                                cw(j, j, Adiag(j) - M * gbar(j) * gbar(j)), 1e-300));
      const double cij = ew(i, j) / std::sqrt(std::max(ew(i, i) * ew(j, j), 1e-300));
      if ((bij - cij) / 2.0 > params.king_screen) pass_ea.push_back(ce[c]);
    }
    const std::vector<double> k0 = dwg_k0(data, params, F.axes, Mat2D(), pass_ea);
    std::vector<std::pair<int, int>> add;
    for (size_t c = 0; c < pass_ea.size(); ++c)
      if (k0[c] < DWG_K0_MAX) add.push_back(pass_ea[c]);
    n_ea = add.size();
    if (add.empty()) break;  // nothing new
    for (const auto& pr : add) used.insert(pr);
    std::vector<std::pair<int, int>> c2(used.begin(), used.end());
    add_cands(c2);
    F = fit(c2);
    std::set<std::pair<int, int>> nk;
    for (const auto& [i, j, phi] : F.pairs) nk.emplace(i, j);
    if (nk == key) break;
    key = nk;
  }
  const double t_screen = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts0).count();
  cao.print(tick.date(), "robust PCA (dwg): first fit", t_fit, "s (", passes_fit, "passes ), screen", t_screen, "s");
  cao.print(tick.date(), "robust PCA (dwg): GRM-scale fit, rank", F.r, "(at most", k + 1, ", noise edge", edge, "),",
            F.pairs.size(), "related pairs;", n_ea, "candidates added by evalAdmix + k0 in the last round;", rounds + 1,
            "screen rounds;", passes, "passes");
  // whitening of A_s with the fitted noise
  BlockDiag Wm, Wh;
  build_whitener(F.v, F.pairs, Wm, Wh);
  // centred in the whitened space: project out u = Sigma^-1/2 1 (the mean
  // direction), so the top k eigenvectors are the PCs (see whitened_cs)
  Mat1D u = Wm.apply(Mat2D::Ones(N, 1)).col(0);
  u.normalize();
  auto proj = [&](const Mat2D& X) { return Mat2D(X - u * (u.transpose() * X)); };
  auto wop = [&](const Mat2D& X) { return proj(Wm.apply(As_apply(Wm.apply(proj(X))))); };
  Eigen::HouseholderQR<Mat2D> qr0(proj(Wm.apply(V)));
  Mat2D V0 = qr0.householderQ() * Mat2D::Identity(N, V.cols());
  Mat1D theta, res;
  Mat2D Vr;
  double resid = 1;
  int extra = -1;
  for (int it = 0; it < 5000; ++it) {
    ritz_step(wop, V0, theta, Vr, resid, std::max(1, F.r - 1), &res);
    if (resid < conv_tol && extra < 0) extra = 0;
    if (resid < conv_tol && (res.head(k).maxCoeff() < std::max(conv_tol, 1e-6) || extra++ >= 10)) break;
  }
  const Mat2D U = Wh.apply(Vr.leftCols(k));
  const Mat1D w = theta.head(k);
  cao.print(tick.date(), "robust PCA (dwg):", passes, "passes over the genotypes in total");
  write_outputs(data, params, N, U, w, F.pairs, [&](int i, int j) {
    auto it = king_map.find({std::min(i, j), std::max(i, j)});
    return it == king_map.end() ? NAN : it->second;
  });
}


// ---- dwg with genotype likelihoods (BEAGLE input, dense engine) ---------------
// (scripts/relatepca/dwg_gl.py, deshrink variant). Posteriors with the sample
// allele frequencies as prior; the posterior means E shrink towards the prior
// by the individual's genotype reliability rho_i = 1 - sum Var(g|data) /
// sum 2f(1-f), so they are deshrunk, x = 2f + (E - 2f) / rho_i, which makes
// the off-diagonal cross-products approximately unbiased (the extra noise
// goes to the free diagonal). The kinship rule uses the noise times rho
// (phi / sqrt(rho_i rho_j)); the whitening uses the observed covariance.
// Candidates: KING from expected counts (posterior probabilities) and the
// evalAdmix screen on x, both corrected by sqrt(rho_i rho_j); evalAdmix-only
// candidates are confirmed by k0 < 0.8 from the genotype likelihoods. Final
// relatedness: (k0, k1, k2) by maximum likelihood from the likelihoods with
// leave-the-family-out allele frequencies from the final PCs.

// per-site IBD likelihoods of a pair from genotype likelihoods (allele-level
// model of pcaone-ibd summed over the genotypes); gi, gj: (GL0, GL1, GL2)
static inline void gl_pair_lik(const double* gi, const double* gj, double pa, double pb, double* l) {
  const double qa = 1 - pa, qb = 1 - pb, ps = 0.5 * (pa + pb), qs = 1 - ps;
  const double ha[3] = {qa * qa, 2 * pa * qa, pa * pa}, hb[3] = {qb * qb, 2 * pb * qb, pb * pb};
  const double hs[3] = {qs * qs, 2 * ps * qs, ps * ps};
  double ai = 0, aj = 0, l2 = 0;
  for (int g = 0; g < 3; ++g) {
    ai += gi[g] * ha[g];
    aj += gj[g] * hb[g];
    l2 += gi[g] * gj[g] * hs[g];
  }
  // IBD1: shared allele S; P(g | S) = P(other allele = g - S)
  const double i0 = gi[0] * qa + gi[1] * pa, i1 = gi[1] * qa + gi[2] * pa;
  const double j0 = gj[0] * qb + gj[1] * pb, j1 = gj[1] * qb + gj[2] * pb;
  l[0] = ai * aj;
  l[1] = qs * i0 * j0 + ps * i1 * j1;
  l[2] = l2;
}

static void run_dwg_gl(Data* data, const Param& params) {
  const Eigen::Index N = data->nsamples, M = data->nsnps;
  const int k = params.k;
  const double tau = params.kin_min;
  cao.print(tick.date(), "robust PCA ( dwg, genotype likelihoods ): N =", N, ", M =", M, ", k =", k);
  tick.clock();
  auto col = [&](Eigen::Index j) { return (Eigen::Index)(params.filterSNP ? data->keepSNPs[j] : j); };
  const Mat2D& P = data->P;
  // genotype likelihoods as N x M x 3 (contiguous per individual-site)
  std::vector<double> GL(N * M * 3);
#pragma omp parallel for num_threads(params.threads)
  for (Eigen::Index j = 0; j < M; ++j) {
    const Eigen::Index s = col(j);
    for (Eigen::Index i = 0; i < N; ++i) {
      double* g = &GL[(i * M + j) * 3];
      g[0] = P(2 * i, s);
      g[1] = P(2 * i + 1, s);
      g[2] = std::max(0.0, 1.0 - g[0] - g[1]);
    }
  }
  // allele frequencies by EM from 0.25 (as emMAF_with_GL; data->F is not
  // estimated here, because PCAone sets maxiter = 0 outside EMU/PCAngsd mode)
  Mat1D f = Mat1D::Constant(M, 0.25);
  for (int it = 0; it < 100; ++it) {
    Mat1D fn(M);
#pragma omp parallel for num_threads(params.threads)
    for (Eigen::Index j = 0; j < M; ++j) {
      const double q = f(j), h0 = (1 - q) * (1 - q), h1 = 2 * q * (1 - q), h2 = q * q;
      double t = 0;
      for (Eigen::Index i = 0; i < N; ++i) {
        const double* g = &GL[(i * M + j) * 3];
        const double a = g[0] * h0, b = g[1] * h1, c = g[2] * h2;
        t += (b + 2 * c) / (a + b + c);
      }
      fn(j) = t / (2.0 * N);
    }
    const double d = std::sqrt((fn - f).squaredNorm() / M);
    f = fn;
    if (d < 1e-6) break;
  }
  f = f.cwiseMax(1e-4).cwiseMin(1 - 1e-4);
  // posteriors with the prior f: E[g], probabilities, and the reliability
  Mat2D E(N, M), P0(N, M), P1(N, M), P2(N, M);
  Mat1D varsum = Mat1D::Zero(N), priorsum = Mat1D::Zero(N);
#pragma omp parallel num_threads(params.threads)
  {
    Mat1D vs = Mat1D::Zero(N), ps = Mat1D::Zero(N);
#pragma omp for
    for (Eigen::Index j = 0; j < M; ++j) {
      const double q = f(j), h0 = (1 - q) * (1 - q), h1 = 2 * q * (1 - q), h2 = q * q;
      for (Eigen::Index i = 0; i < N; ++i) {
        const double* g = &GL[(i * M + j) * 3];
        double a = g[0] * h0, b = g[1] * h1, c = g[2] * h2;
        const double t = a + b + c;
        a /= t;
        b /= t;
        c /= t;
        P0(i, j) = a;
        P1(i, j) = b;
        P2(i, j) = c;
        E(i, j) = b + 2 * c;
        vs(i) += b + 4 * c - E(i, j) * E(i, j);
        ps(i) += h1;
      }
    }
#pragma omp critical
    {
      varsum += vs;
      priorsum += ps;
    }
  }
  const Mat1D rho = (Mat1D::Ones(N) - varsum.cwiseQuotient(priorsum)).cwiseMax(0.02).cwiseMin(1.0);
  // deshrunk posterior means, the GRM-scale Gram, expected KING counts
  Mat2D X(N, M);
  for (Eigen::Index j = 0; j < M; ++j) X.col(j) = (2 * f(j) + (E.col(j).array() - 2 * f(j)) / rho.array()).matrix();
  const Mat1D w = (2 * f.array() * (1 - f.array())).matrix();
  const Mat2D Xs = X * w.cwiseSqrt().cwiseInverse().asDiagonal();
  const Mat2D As = Xs * Xs.transpose() / (double)M;
  const Mat2D Araw = X * X.transpose();
  const Mat1D gbar = X.rowwise().mean();
  const Mat1D dx = (X.array() * (2.0 - X.array())).rowwise().mean().matrix();
  double edge_sum = 0;
  for (Eigen::Index j = 0; j < M; ++j) {
    const double m1 = X.col(j).mean(), m2 = X.col(j).squaredNorm() / N;
    edge_sum += std::max(m2 * m2 - m1 * m1 * m1 * m1, 0.0) / (w(j) * w(j));
  }
  const double edge = 2.0 * std::sqrt(edge_sum / ((double)M * M)) * std::sqrt((double)N);
  const Mat2D HH = P1 * P1.transpose(), I0 = P0 * P2.transpose() + P2 * P0.transpose();
  const Mat1D nhet = P1.rowwise().sum();
  Mat2D king(N, N);
  for (Eigen::Index j = 0; j < N; ++j)
    for (Eigen::Index i = 0; i < N; ++i) {
      const double mn = std::max(std::min(nhet(i), nhet(j)), 1e-9);
      king(i, j) = ((HH(i, j) - 2.0 * I0(i, j)) / (2.0 * mn) + 0.5 - (nhet(i) + nhet(j)) / (4.0 * mn)) /
                   std::sqrt(rho(i) * rho(j));
    }
  king.diagonal().setZero();
  MatB cand = (king.array() > params.king_screen).matrix();
  cao.print(tick.date(), "robust PCA (dwg, GL): posteriors and summaries in", tick.reltime(),
            "seconds; mean genotype reliability", rho.mean());

  // IAF of individual a and b at every site, given axes, leaving out `out`
  auto iaf_pair = [&](const Mat2D& axes, int a, int b, const std::vector<int>& out, Mat1D& pa, Mat1D& pb) {
    Mat2D V(N, axes.cols() + 1);
    V.leftCols(axes.cols()) = axes;
    V.col(axes.cols()).setOnes();
    Mat2D VtV = V.transpose() * V;
    Mat2D R = V.transpose() * X;
    for (int t : out) {
      VtV.noalias() -= V.row(t).transpose() * V.row(t);
      R.noalias() -= V.row(t).transpose() * X.row(t);
    }
    const Mat2D Ai = VtV.completeOrthogonalDecomposition().pseudoInverse();
    pa = (0.5 * (V.row(a) * Ai * R)).transpose();
    pb = (0.5 * (V.row(b) * Ai * R)).transpose();
    pa = pa.cwiseMax(1e-3).cwiseMin(1 - 1e-3);
    pb = pb.cwiseMax(1e-3).cwiseMin(1 - 1e-3);
  };
  auto pair_ibd = [&](int a, int b, const Mat1D& pa, const Mat1D& pb, double* kk, bool exact) {
    std::vector<double> lik(3 * M);
    for (Eigen::Index j = 0; j < M; ++j)
      gl_pair_lik(&GL[(a * M + j) * 3], &GL[(b * M + j) * 3], pa(j), pb(j), &lik[3 * j]);
    if (exact) {
      kk[0] = kk[1] = kk[2] = 1.0 / 3.0;
      ml_ibd(lik, M, kk);
    } else {  // screen: 30 EM iterations are enough to decide k0 < 0.8
      double c[3] = {1.0 / 3, 1.0 / 3, 1.0 / 3};
      for (int it = 0; it < 30; ++it) {
        double s0 = 0, s1 = 0, s2 = 0;
        for (Eigen::Index j = 0; j < M; ++j) {
          const double w0 = c[0] * lik[3 * j], w1 = c[1] * lik[3 * j + 1], w2 = c[2] * lik[3 * j + 2], t = w0 + w1 + w2;
          if (t > 0) {
            s0 += w0 / t;
            s1 += w1 / t;
            s2 += w2 / t;
          }
        }
        const double tot = s0 + s1 + s2, d = std::fabs(s0 / tot - c[0]);
        c[0] = s0 / tot;
        c[1] = s1 / tot;
        c[2] = s2 / tot;
        if (d < 1e-4) break;
      }
      std::copy(c, c + 3, kk);
    }
  };

  DwgFit F = dwg_fit(As, tau, cand, edge, k + 1, 500, rho);
  MatB used = cand;
  std::set<std::pair<int, int>> key;
  for (const auto& [i, j, phi] : F.pairs) key.emplace(i, j);
  int rounds = 0, n_ea = 0;
  for (; rounds < 3; ++rounds) {
    const Mat2D ea = dwg_evaladmix(Araw, gbar, dx, (double)M, F.axes);
    std::vector<std::pair<int, int>> ce;
    for (Eigen::Index j = 0; j < N; ++j)
      for (Eigen::Index i = 0; i < j; ++i)
        if (ea(i, j) / std::sqrt(rho(i) * rho(j)) > params.king_screen && !used(i, j)) ce.emplace_back(i, j);
    std::vector<char> ok(ce.size(), 0);
#pragma omp parallel for num_threads(params.threads) schedule(dynamic)
    for (size_t c = 0; c < ce.size(); ++c) {
      Mat1D pa, pb;
      iaf_pair(F.axes, ce[c].first, ce[c].second, {}, pa, pb);
      double kk[3];
      pair_ibd(ce[c].first, ce[c].second, pa, pb, kk, false);
      ok[c] = kk[0] < DWG_K0_MAX;
    }
    MatB c2 = used;
    n_ea = 0;
    for (size_t c = 0; c < ce.size(); ++c)
      if (ok[c]) {
        c2(ce[c].first, ce[c].second) = c2(ce[c].second, ce[c].first) = true;
        ++n_ea;
      }
    if (c2 == used) break;
    used = c2;
    F = dwg_fit(As, tau, c2, edge, k + 1, 500, rho);
    std::set<std::pair<int, int>> nk;
    for (const auto& [i, j, phi] : F.pairs) nk.emplace(i, j);
    if (nk == key) break;
    key = nk;
  }
  cao.print(tick.date(), "robust PCA (dwg, GL): rank", F.r, "(at most", k + 1, ", noise edge", edge, "),",
            F.pairs.size(), "related pairs;", n_ea, "candidates added by evalAdmix + k0;", rounds + 1, "screen rounds");
  const Mat1D noise = (As.diagonal() - F.L.diagonal()).cwiseMax(1e-6);
  Mat1D wv;
  Mat2D U;
  whitened_cs(As, noise, F.pairs, k, wv, U, F.r, true);

  // final relatedness from the genotype likelihoods, leave-the-family-out IAF
  std::vector<int> root(N);
  std::iota(root.begin(), root.end(), 0);
  std::function<int(int)> find = [&](int x) { return root[x] == x ? x : root[x] = find(root[x]); };
  for (const auto& [i, j, phi] : F.pairs) root[find(i)] = find(j);
  std::map<int, std::vector<int>> family;
  std::vector<int> fam_of(N);
  for (Eigen::Index i = 0; i < N; ++i) family[fam_of[i] = find(i)].push_back(i);
  std::vector<FinalRel> fin(F.pairs.size());
  std::vector<Pair> pairs_out(F.pairs.size());
#pragma omp parallel for num_threads(params.threads) schedule(dynamic)
  for (size_t c = 0; c < F.pairs.size(); ++c) {
    const auto& [i, j, phi] = F.pairs[c];
    std::vector<int> out = family.at(fam_of[i]);
    if ((Eigen::Index)out.size() > 100) out = {i, j};
    Mat1D pa, pb;
    iaf_pair(U, i, j, out, pa, pb);
    double kk[3];
    pair_ibd(i, j, pa, pb, kk, true);
    fin[c].k0 = kk[0];
    fin[c].k1 = kk[1];
    fin[c].k2 = kk[2];
    fin[c].kin = 0.25 * kk[1] + 0.5 * kk[2];
    // the detection kinship, corrected for the genotype reliability
    pairs_out[c] = Pair(i, j, phi / std::sqrt(rho(i) * rho(j)));
  }
  cao.print(tick.date(), "robust PCA (dwg, GL): final kinship and IBD sharing (k0, k1, k2) from the genotype likelihoods");
  write_outputs(data, params, N, U, wv, pairs_out, [&](int i, int j) { return king(i, j); }, &fin);
}

void run_robust(Data* data, const Param& params) {
  if (params.file_t == FileType::BEAGLE) {
    if (params.robust != "dwg") cao.error("--robust with genotype likelihoods (-G) supports dwg only");
    return run_dwg_gl(data, params);
  }
  bool op = params.robust != "aarobust-kin" && params.robust != "diag-impute" &&
            (params.robust_engine == "operator" || params.robust_engine == "iram" ||
             (params.robust_engine == "auto" && data->nsamples > (uint)params.robust_dense_max));
  // dwg: the operator engine is faster at every N in-core (N = 1000: 1.1 s
  // against 4.5 s); out-of-core the dense engine reads the file once
  if (params.robust == "dwg" && params.robust_engine == "auto")
    op = !params.out_of_core || data->nsamples > (uint)params.robust_dense_max;
  if (op && params.robust == "dwg") return run_dwg_operator(data, params);
  if (op) return run_robust_operator(data, params);
  const Eigen::Index N = data->nsamples;
  const int k = params.k;
  const double tau = params.kin_min;
  const std::string& mode = params.robust;
  const bool need_grm = mode == "aarobust-kin" || mode == "diag-impute";
  const bool need_dwg = mode == "dwg";
  cao.print(tick.date(), "robust PCA (", mode, "): N =", N, ", k =", k, ", tau =", tau,
            ", KING screen =", params.king_screen);

  // ---- 1. one streaming pass ---------------------------------------------
  Summaries s;
  s.A = s.HH = s.I0 = Mat2D::Zero(N, N);
  if (need_grm) s.C = Mat2D::Zero(N, N);
  if (need_dwg) s.As = Mat2D::Zero(N, N);
  s.Dsum = s.nhet = s.gsum = Mat1D::Zero(N);
  tick.clock();
  const Eigen::Index bsz = 4096;
  if (!params.out_of_core) {
    // in-core: raw {0, 0.5, 1} with -9 for missing (params.center is false)
    const Mat2D& G = data->G;
    if ((G.array() < 0).any()) cao.error("--robust requires complete genotypes (no missing values)");
    for (Eigen::Index s0 = 0; s0 < G.cols(); s0 += bsz) {
      const Eigen::Index b = std::min(bsz, G.cols() - s0);
      accumulate(G.middleCols(s0, b), data->F.segment(s0, b), need_grm, s, need_dwg);
    }
  } else {
    // out-of-core: the block reader estimates F on the fly and returns
    // centred genotypes regardless of params.center, so add f back (as in
    // run_evaladmix)
    data->F = Mat1D::Zero(data->nsnps);
    data->centered_geno_lookup = Arr2D::Zero(4, data->nsnps);
    data->check_file_offset_first_var();
    for (uint bi = 0; bi < data->nblocks; ++bi) {
      data->read_block_initial(data->start[bi], data->stop[bi], false);
      Mat2D X = data->G;
      Mat1D f = data->F.segment(data->start[bi], X.cols());
      for (Eigen::Index j = 0; j < X.cols(); ++j) X.col(j).array() += f(j);
      accumulate(X, f, need_grm, s, need_dwg);
    }
  }
  const double M = (double)s.M;
  Mat1D D = s.Dsum / M;
  cao.print(tick.date(), "robust PCA: accumulated", s.M, "sites in", tick.reltime(), "seconds");

  // ---- 2. KING-robust kinship (plink2 --make-king) and the screen ---------
  Mat2D king(N, N);
  for (Eigen::Index j = 0; j < N; ++j)
    for (Eigen::Index i = 0; i < N; ++i) {
      const double mn = std::max(std::min(s.nhet(i), s.nhet(j)), 1.0);
      king(i, j) = (s.HH(i, j) - 2.0 * s.I0(i, j)) / (2.0 * mn) + 0.5 - (s.nhet(i) + s.nhet(j)) / (4.0 * mn);
    }
  king.diagonal().setZero();
  MatB cand = (king.array() > params.king_screen).matrix();
  if (!params.filekin.empty() && mode != "cswhite") {
    // candidates from the --kinship pair table instead of the internal KING
    cand.setConstant(false);
    for (const auto& [i, j, phi] : read_kinship_pairs(params, N, params.king_screen))
      if (phi > params.king_screen) cand(i, j) = cand(j, i) = true;
  }

  // ---- 3. the chosen method ----------------------------------------------
  Mat2D AM = s.A / M;
  Mat1D w;
  Mat2D U;
  std::vector<Pair> pairs;
  if (mode == "detect-white" || mode == "frkin") {
    // fixed rank + kinship on the raw Gram with the diagonal free (the CS
    // matrix A/M - D differs only on the diagonal, which the fit does not use);
    // D only scales the residual to kinship
    Mat2D L, S;
    int rank = k + 1;
    // detect-white: rank from the noise edge, at most k + 1 (a too large -k is
    // reduced; frkin takes its PCs from L and keeps rank k + 1)
    const double edge = mode == "detect-white" && !params.robust_fixed_rank ? noise_edge(s.edge_sum, M, N) : 0.0;
    int it = lr_kin(AM, rank, tau, D, cand, L, S, true, edge);
    pairs = lr_kin_pairs(AM, D, L, S);
    cao.print(tick.date(), "robust PCA: fixed rank + kinship fit, rank", rank, "(at most", k + 1, ", noise edge", edge, "),", it, "iterations,", pairs.size(),
              "related pairs detected");
    if (mode == "frkin") {
      cs_pcs(L, k, w, U);
    } else {
      // whitening with the estimated noise variance: observed diagonal minus structure
      Mat1D noise = AM.diagonal() - L.diagonal();
      whitened_cs(AM, noise, pairs, k, w, U, rank);
    }
  } else if (mode == "dwg") {
    const Mat2D As = s.As / (double)s.Ms;
    const double edge = 2.0 * std::sqrt(s.edge_sum_s / ((double)s.Ms * s.Ms)) * std::sqrt((double)N);
    const Mat1D gbar = s.gsum / M;
    MatB off = MatB::Constant(N, N, true);
    off.diagonal().setConstant(false);
    DwgFit F = dwg_fit(As, tau, cand, edge, k + 1);
    MatB used = cand;  // candidates of the current fit
    std::set<std::pair<int, int>> key;  // pairs of the current fit: stop when a refit does not change them
    for (const auto& [i, j, phi] : F.pairs) key.emplace(i, j);
    int rounds = 0, n_ea = 0;
    for (; rounds < 5; ++rounds) {
      // evalAdmix candidates from the structure axes, confirmed by k0
      const Mat2D ea = dwg_evaladmix(s.A, gbar, D, M, F.axes);
      std::vector<std::pair<int, int>> ce;
      for (Eigen::Index j = 0; j < N; ++j)
        for (Eigen::Index i = 0; i < j; ++i)
          if (ea(i, j) > params.king_screen && !cand(i, j)) ce.emplace_back(i, j);
      const std::vector<double> k0 = dwg_k0(data, params, F.axes, s.I0, ce);
      MatB c2 = cand;
      n_ea = 0;
      for (size_t c = 0; c < ce.size(); ++c)
        if (k0[c] < DWG_K0_MAX) {
          c2(ce[c].first, ce[c].second) = c2(ce[c].second, ce[c].first) = true;
          ++n_ea;
        }
      if (c2 == used) break;  // nothing new: the fit would be the same
      used = c2;
      F = dwg_fit(As, tau, c2, edge, k + 1);
      std::set<std::pair<int, int>> nk;
      for (const auto& [i, j, phi] : F.pairs) nk.emplace(i, j);
      if (nk == key) break;
      key = nk;
    }
    pairs = F.pairs;
    cao.print(tick.date(), "robust PCA (dwg): GRM-scale fit, rank", F.r, "(at most", k + 1, ", noise edge", edge,
              "),", pairs.size(), "related pairs;", n_ea, "candidates added by evalAdmix + k0;", rounds + 1,
              "screen rounds");
    const Mat1D noise = (As.diagonal() - F.L.diagonal()).cwiseMax(1e-6);
    whitened_cs(As, noise, pairs, k, w, U, F.r, true);
  } else if (mode == "cswhite") {
    cao.warn("--robust cswhite corrects the diagonal with heterozygosity, which assumes HWE within "
             "individuals; use detect-white with inbreeding or genotype errors");
    if (!params.filekin.empty()) {
      KinshipWhitener kw(params, N);  // reads and thresholds the pair table / matrix
      pairs = kw.close_pairs();
    } else {
      for (Eigen::Index j = 0; j < N; ++j)
        for (Eigen::Index i = 0; i < j; ++i)
          if (king(i, j) >= tau) pairs.emplace_back(i, j, king(i, j));
    }
    cao.print(tick.date(), "robust PCA: CS whitening with", pairs.size(), "related pairs");
    whitened_cs(AM, D, pairs, k, w, U);
  } else if (mode == "aarobust-kin") {
    Mat2D C = s.C / M, S;
    const double t_edge = 1.1 * 2.0 * std::sqrt((double)N / M) * C.diagonal().mean();
    Mat2D L = params.pcp_edge ? pcp_kin_soft(C, tau, cand, S, t_edge, params.pcp_iram)
                              : pcp_kin(C, tau, cand, S, params.pcp_iram);
    if (params.impute_diag) {
      top_eig(L, k, w, U);  // the fitted structure, diagonal imputed
    } else {
      // the GRM with the related pairs replaced by the fit, observed diagonal
      // kept: standard PCA as if the relatives were unrelated
      Mat2D Cu = C - S;
      Cu.diagonal() = C.diagonal();
      top_eig(Cu, k, w, U);
    }
    Mat1D noise = (C - L).diagonal();
    for (Eigen::Index j = 0; j < N; ++j)
      for (Eigen::Index i = 0; i < j; ++i)
        if (std::abs(S(i, j)) > 1e-12)
          pairs.emplace_back(i, j, S(i, j) / (2.0 * std::sqrt(std::max(noise(i) * noise(j), 1e-24))));
    cao.print(tick.date(), "robust PCA: PCP with kinship threshold,", pairs.size(), "related pairs detected;",
              params.impute_diag ? "PCs of the fit (diagonal imputed)" : "PCs of the GRM without the related pairs");
  } else if (mode == "diag-impute") {
    // PCA of unrelated samples with the GRM diagonal imputed from the
    // off-diagonal entries (PCP with only the diagonal unobserved; no pairs).
    // At small n the observed diagonal reflects heterozygosity, not structure.
    Mat2D C = s.C / M, S;
    MatB none = MatB::Constant(N, N, false);
    const double t_edge = 1.1 * 2.0 * std::sqrt((double)N / M) * C.diagonal().mean();
    Mat2D L = params.pcp_edge ? pcp_kin_soft(C, tau, none, S, t_edge, params.pcp_iram)
                              : pcp_kin(C, tau, none, S, params.pcp_iram);
    top_eig(L, k, w, U);
    cao.print(tick.date(), "robust PCA: GRM with the diagonal imputed (no related pairs considered)");
  } else {
    cao.error("unknown --robust mode " + mode);
  }

  // ---- 4. output (diag-impute: --impute-diag without --robust) ---------------------------------------------------------
  write_outputs(data, params, N, U, w, pairs, [&](int i, int j) { return king(i, j); });
}
