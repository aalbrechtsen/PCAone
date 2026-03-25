/*******************************************************************************
 * @file        https://github.com/Zilong-Li/PCAone/src/Projection.cpp
 * @author      Zilong Li
 * Copyright (C) 2024. Use of this code is governed by the LICENSE file.
 ******************************************************************************/

#include "Projection.hpp"

#include "Utils.hpp"

static void solve_projection_scores(const Mat2D& V, const ArrBool& C, const Mat2D& G, Mat2D& U) {
  if (U.rows() == 0 || U.cols() == 0) return;
  double p_miss = C.size() ? (double)C.count() / (double)C.size() : 0.0;
  if (p_miss == 0.0) {
    Eigen::ColPivHouseholderQR<Mat2D> qr(V);
#pragma omp parallel for
    for (uint i = 0; i < (uint)U.rows(); i++) {
      U.row(i) = qr.solve(G.row(i).transpose());
    }
  } else {
#pragma omp parallel for
    for (uint i = 0; i < (uint)U.rows(); i++) {
      Int1D idx;
      for (int j = 0; j < V.rows(); j++) {
        if (!C(j * U.rows() + i)) idx.push_back(j);
      }
      U.row(i) = V(idx, Eigen::all).colPivHouseholderQr().solve(G(i, idx).transpose());
    }
  }
}

static void run_projection_beagle_gl(Data* data, const Param& params, const Mat1D& S, const Mat2D& Vref) {
  const int pcs = S.size();
  Mat2D Gproj(data->nsamples, data->project_overlap);
  ArrBool Cproj = ArrBool::Zero(data->project_overlap * data->nsamples);
  Mat1D Fproj(data->project_overlap);
  Mat2D Pproj(data->nsamples * 2, data->project_overlap);
  Mat2D L(data->project_overlap, pcs);
  uint matched_idx = 0;
  for (uint j = 0; j < data->nsnps; ++j) {
    int ref_idx = data->project_ref_indices[j];
    if (ref_idx < 0) continue;
    Gproj.col(matched_idx) = data->G.col(j);
    Fproj(matched_idx) = data->F(j);
    Pproj.col(matched_idx) = data->P.col(j);
    for (uint i = 0; i < data->nsamples; ++i) {
      Cproj[matched_idx * data->nsamples + i] = data->C.size() ? data->C[j * data->nsamples + i] : false;
    }
    L.row(matched_idx) = Vref.row(ref_idx);
    if (data->project_flip[j]) L.row(matched_idx) *= -1.0;
    matched_idx++;
  }

  data->G = Gproj;
  data->C = Cproj;
  data->F = Fproj;
  data->P = Pproj;
  data->nsnps = data->project_overlap;

  Mat2D VS = L * S.asDiagonal();
  Mat2D U = Mat2D::Zero(data->nsamples, pcs);

  // Initial posterior mean under the reference AF prior.
#pragma omp parallel for
  for (uint j = 0; j < data->nsnps; ++j) {
    const double norm = sqrt(2.0 * data->F(j) * (1.0 - data->F(j)));
    for (uint i = 0; i < data->nsamples; ++i) {
      double pt = fmin(fmax(data->F(j), 1e-4), 1.0 - 1e-4);
      double gl11 = data->P(2 * i + 0, j);
      double gl12 = data->P(2 * i + 1, j);
      double gl22 = 1.0 - gl11 - gl12;
      double p11 = gl11 * pt * pt;
      double p12 = gl12 * 2.0 * pt * (1.0 - pt);
      double p22 = gl22 * (1.0 - pt) * (1.0 - pt);
      double pSum = p11 + p12 + p22;
      if (!std::isfinite(pSum) || pSum <= 0.0) {
        data->C[j * data->nsamples + i] = 1;
        data->G(i, j) = 0.0;
        continue;
      }
      data->C[j * data->nsamples + i] = 0;
      data->G(i, j) = (2.0 * p11 + p12) / (2.0 * pSum) - data->F(j);
      if (params.standardize_geno && norm > VAR_TOL) data->G(i, j) /= norm;
    }
  }

  solve_projection_scores(VS, data->C, data->G, U);

  for (uint iter = 0; iter < params.maxiter; ++iter) {
    Mat2D Uprev = U;
#pragma omp parallel for
    for (uint j = 0; j < data->nsnps; ++j) {
      const double norm = sqrt(2.0 * data->F(j) * (1.0 - data->F(j)));
      for (uint i = 0; i < data->nsamples; ++i) {
        double z = 0.0;
        for (int k = 0; k < pcs; ++k) {
          z += U(i, k) * S(k) * L(j, k);
        }
        double centered = z;
        if (params.standardize_geno && norm > VAR_TOL) centered *= norm;
        double pt = centered + data->F(j);
        pt = fmin(fmax(pt, 1e-4), 1.0 - 1e-4);

        double gl11 = data->P(2 * i + 0, j);
        double gl12 = data->P(2 * i + 1, j);
        double gl22 = 1.0 - gl11 - gl12;
        double p11 = gl11 * pt * pt;
        double p12 = gl12 * 2.0 * pt * (1.0 - pt);
        double p22 = gl22 * (1.0 - pt) * (1.0 - pt);
        double pSum = p11 + p12 + p22;
        if (!std::isfinite(pSum) || pSum <= 0.0) {
          data->C[j * data->nsamples + i] = 1;
          data->G(i, j) = 0.0;
          continue;
        }
        data->C[j * data->nsamples + i] = 0;
        data->G(i, j) = (2.0 * p11 + p12) / (2.0 * pSum) - data->F(j);
        if (params.standardize_geno && norm > VAR_TOL)
          data->G(i, j) /= norm;
        else if (params.standardize_geno)
          data->G(i, j) = 0.0;
      }
    }

    solve_projection_scores(VS, data->C, data->G, U);

    for (int k = 0; k < pcs; ++k) {
      if (U.col(k).dot(Uprev.col(k)) < 0.0) U.col(k) *= -1.0;
    }

    double denom = Uprev.norm();
    if (denom < 1e-12) denom = 1.0;
    double diff = (U - Uprev).norm() / denom;
    cao.print(tick.date(), "GL projection iter =", iter + 1, ", diff =", diff);
    if (diff < params.tolem) break;
  }

  data->write_eigs_files(S.array().square() / data->nsnps, S, U, VS);
}

/**
 * options:
 * 1: simple, assume no missingness
 * 2: like smartPCA, solving g=Vx, can take missing genotypes
 * 3: OADP, laser2, can take missing genotypes
 */
void run_projection(Data* data, const Param& params) {
  cao.print(tick.date(), "run projection");

  data->prepare();
  uint ref_nsamples = 0, ref_nsnps = 0;
  Mat1D S;
  read_sigvals(params.fileS, ref_nsamples, ref_nsnps, S);
  if (data->project_overlap == 0) {
    cao.error("no overlapping SNPs available for projection");
  }
  Mat2D Vref = read_eigvecs(params.fileV, ref_nsnps, S.size());  // ref_nsnps x K
  if (params.file_t == FileType::BEAGLE) {
    run_projection_beagle_gl(data, params, S, Vref);
    return;
  }
  Mat2D Gproj(data->nsamples, data->project_overlap);
  ArrBool Cproj = ArrBool::Zero(data->project_overlap * data->nsamples);
  Mat1D Fproj(data->project_overlap);
  uint matched_idx = 0;
  for (uint j = 0; j < data->nsnps; ++j) {
    if (data->project_ref_indices[j] < 0) continue;
    Gproj.col(matched_idx) = data->G.col(j);
    Fproj(matched_idx) = data->F(j);
    for (uint i = 0; i < data->nsamples; ++i) {
      Cproj[matched_idx * data->nsamples + i] = data->C.size() ? data->C[j * data->nsamples + i] : false;
    }
    matched_idx++;
  }
  data->G = Gproj;
  data->C = Cproj;
  data->F = Fproj;
  data->nsnps = data->project_overlap;
  if (params.standardize_geno) data->standardize_E();
  double p_miss = data->C.size() ? (double)data->C.count() / (double)data->C.size() : 0.0;
  const int pcs = S.size();
  Mat2D V(data->nsnps, pcs);
  matched_idx = 0;
  for (uint j = 0; j < (uint)data->project_ref_indices.size(); ++j) {
    int ref_idx = data->project_ref_indices[j];
    if (ref_idx < 0) continue;
    V.row(matched_idx) = Vref.row(ref_idx);
    if (data->project_flip[j]) V.row(matched_idx) *= -1.0;
    matched_idx++;
  }
  Mat2D U(data->nsamples, pcs);

  if (params.project == 1) {
    if (p_miss > 0) cao.warn("there are missing genotypes. recommend using --project 2 or 3.");
    // get 1 / Singular = sqrt(Eigen * M)
    V = V * (S.array().inverse().matrix().asDiagonal());
    // G V = U D
    U = data->G * V;
  } else if (params.project == 2) {
    V = V * S.asDiagonal();
    if (p_miss == 0.0) cao.warn("there is no missing genotypes");
    solve_projection_scores(V, data->C, data->G, U);
  } else {
    cao.error("have not implemented yet");
  }

  data->write_eigs_files(S.array().square() / data->nsnps, S, U, V);
}
