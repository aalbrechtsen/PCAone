/*******************************************************************************
 * @file        https://github.com/Zilong-Li/PCAone/src/Kinship.cpp
 * @author      Anders Albrechtsen
 * Copyright (C) 2026. Use of this code is governed by the LICENSE file.
 *
 * Kinship-whitened PCA, see Kinship.hpp for the model.
 *
 * Two kinship file formats are accepted:
 *
 *  1. a pair table with a header naming an ID column for each member and a
 *     kinship column, e.g. pcaone-ibd .ibd (ID1 ID2 kin), KING / PLINK2 .kin0
 *     (FID1 ID1/IID1 FID2 ID2/IID2 ... KINSHIP). Matching is case-insensitive.
 *  2. an N x N matrix, e.g. the .kinship written by --evaladmix, optionally
 *     with a header line of sample IDs. Without one, rows are in sample order.
 *
 * IDs are matched against the IID column of the .fam / .psam.
 ******************************************************************************/
#include "Kinship.hpp"

#include <fstream>
#include <map>
#include <sstream>

#include "Utils.hpp"

// floor on the eigenvalues of each Sigma_f. Duplicates / MZ twins (phi = 1/2)
// make Sigma_f singular; noisy kinship estimates can make it indefinite.
static const double KIN_EIG_MIN = 0.1;

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (!s.empty() && s[0] == '#') s.erase(0, 1);
  return s;
}

static bool is_number(const std::string& s) {
  char* end = nullptr;
  std::strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0';
}

String1D read_sample_ids(const Param& params) {
  String1D ids;
  const std::string sep{" \t"};
  std::string line;
  if (params.file_t == FileType::PLINK) {
    std::ifstream ifs(params.filein + ".fam");
    while (std::getline(ifs, line)) {
      auto tok = split_string(line, sep);
      if (tok.size() >= 2) ids.push_back(tok[1]);
    }
  } else if (params.file_t == FileType::PGEN) {
    std::ifstream ifs(params.filein + ".psam");
    int col = 0;
    while (std::getline(ifs, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        if (line.size() > 1 && line[1] == '#') continue;
        auto tok = split_string(line, sep);
        for (size_t c = 0; c < tok.size(); ++c)
          if (lower(tok[c]) == "iid") col = c;
        continue;
      }
      auto tok = split_string(line, sep);
      if ((int)tok.size() > col) ids.push_back(tok[col]);
    }
  }
  return ids;
}

// union-find over samples
static int find_root(std::vector<int>& parent, int i) {
  while (parent[i] != i) i = parent[i] = parent[parent[i]];
  return i;
}

std::vector<std::tuple<int, int, double>> read_kinship_pairs(const Param& params, uint N, double kmin) {
  const std::string sep{" \t,"};
  String1D ids = read_sample_ids(params);
  if (!ids.empty() && ids.size() != N) cao.error("number of sample IDs does not match the number of samples");
  std::unordered_map<std::string, int> id2idx;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!id2idx.emplace(ids[i], i).second) cao.error("duplicated sample ID, can not match --kinship: " + ids[i]);
  }

  std::ifstream ifs(params.filekin);
  if (!ifs.is_open()) cao.error("can not open --kinship file " + params.filekin);
  std::string line;
  if (!std::getline(ifs, line)) cao.error("empty --kinship file " + params.filekin);
  String1D head = split_string(line, sep);

  // close pairs (i < j) with their kinship
  std::map<std::pair<int, int>, double> close;
  uint64 nread = 0, nmissing = 0;
  auto add = [&](int i, int j, double phi) {
    ++nread;
    if (i == j || !(phi >= kmin)) return;
    close[{std::min(i, j), std::max(i, j)}] = phi;
  };

  int c1 = -1, c2 = -1, ck = -1;
  for (size_t c = 0; c < head.size(); ++c) {
    std::string h = lower(head[c]);
    if (c1 < 0 && (h == "id1" || h == "iid1" || h == "id_1")) c1 = c;
    else if (c2 < 0 && (h == "id2" || h == "iid2" || h == "id_2")) c2 = c;
    else if (ck < 0 && (h == "kinship" || h == "kin" || h == "phi")) ck = c;
  }

  if (c1 >= 0 && c2 >= 0 && ck >= 0) {
    if (id2idx.empty()) cao.error("--kinship pair table needs sample IDs from a .fam or .psam");
    cao.print(tick.date(), "read kinship pairs from", params.filekin, "(columns", head[c1], head[c2], head[ck], ")");
    const int need = std::max({c1, c2, ck});
    while (std::getline(ifs, line)) {
      auto tok = split_string(line, sep);
      if ((int)tok.size() <= need) continue;
      auto a = id2idx.find(tok[c1]), b = id2idx.find(tok[c2]);
      if (a == id2idx.end() || b == id2idx.end()) {
        ++nmissing;
        continue;
      }
      add(a->second, b->second, std::strtod(tok[ck].c_str(), nullptr));
    }
    if (nmissing > 0) cao.warn(nmissing, " pairs in --kinship have IDs not in the genotype file; ignored");
  } else {
    // N x N matrix, with or without a header of IDs
    std::vector<int> col2idx(N);
    std::iota(col2idx.begin(), col2idx.end(), 0);
    bool header = !head.empty() && !is_number(head[0]);
    if (header) {
      if (head.size() != N) cao.error("--kinship matrix header does not have N sample IDs");
      if (!id2idx.empty()) {
        for (uint c = 0; c < N; ++c) {
          auto it = id2idx.find(head[c]);
          if (it == id2idx.end()) cao.error("sample in --kinship not found in the genotype file: " + head[c]);
          col2idx[c] = it->second;
        }
      }
    }
    cao.print(tick.date(), "read N x N kinship matrix from", params.filekin, header ? "with" : "without",
              "header");
    uint r = 0;
    auto parse_row = [&](const String1D& tok) {
      if (tok.size() != N) cao.error("row " + std::to_string(r + 1) + " of --kinship matrix does not have N values");
      for (uint c = r + 1; c < N; ++c) add(col2idx[r], col2idx[c], std::strtod(tok[c].c_str(), nullptr));
      ++r;
    };
    if (!header) parse_row(head);
    while (std::getline(ifs, line)) {
      if (line.empty()) continue;
      if (r >= N) cao.error("--kinship matrix has more than N rows");
      parse_row(split_string(line, sep));
    }
    if (r != N) cao.error("--kinship matrix does not have N rows");
  }

  std::vector<std::tuple<int, int, double>> out;
  for (const auto& kv : close) out.emplace_back(kv.first.first, kv.first.second, kv.second);
  cao.print(tick.date(), "--kinship: read", nread, "pairs,", out.size(), "with kinship >=", kmin);
  return out;
}

KinshipWhitener::KinshipWhitener(const Param& params, uint N) {
  const double kmin = params.kin_min;
  pairs_ = read_kinship_pairs(params, N, kmin);
  std::map<std::pair<int, int>, double> close;
  for (const auto& [a, b, phi] : pairs_) close[{a, b}] = phi;
  const size_t nread = pairs_.size();



  // families = connected components of the close-relative graph
  std::vector<int> parent(N);
  std::iota(parent.begin(), parent.end(), 0);
  for (const auto& kv : close) parent[find_root(parent, kv.first.first)] = find_root(parent, kv.first.second);
  std::map<int, Int1D> comp;
  for (uint i = 0; i < N; ++i) comp[find_root(parent, i)].push_back(i);
  for (auto& kv : comp)
    if (kv.second.size() > 1) families.push_back(kv.second);

  size_t nrel = 0, largest = 0;
  int nclipped = 0;
  for (const auto& fam : families) {
    const int n = fam.size();
    nrel += n;
    largest = std::max(largest, fam.size());
    std::unordered_map<int, int> pos;
    for (int a = 0; a < n; ++a) pos[fam[a]] = a;
    Mat2D Sig = Mat2D::Identity(n, n);
    for (int a = 0; a < n; ++a)
      for (int b = a + 1; b < n; ++b) {
        auto it = close.find({fam[a], fam[b]});
        if (it != close.end()) Sig(a, b) = Sig(b, a) = 2.0 * it->second;
      }
    Eigen::SelfAdjointEigenSolver<Mat2D> eig(Sig);
    Mat1D d = eig.eigenvalues();
    for (int a = 0; a < n; ++a)
      if (d(a) < KIN_EIG_MIN) d(a) = KIN_EIG_MIN, ++nclipped;
    const Mat2D& Q = eig.eigenvectors();
    L_half.push_back(Q * d.cwiseSqrt().asDiagonal() * Q.transpose());
    Linv_half.push_back(Q * d.cwiseSqrt().cwiseInverse().asDiagonal() * Q.transpose());
  }
  cao.print(tick.date(), "kinship-whitened PCA: read", nread, "pairs,", close.size(), "with kinship >=", kmin);
  cao.print(tick.date(), "kinship-whitened PCA:", families.size(), "families with", nrel,
            "samples; largest family has", largest, "samples");
  if (nclipped > 0)
    cao.warn(nclipped, " eigenvalue(s) of the family covariance clipped to ", KIN_EIG_MIN,
             " (duplicates or inconsistent kinship)");
  if (families.empty()) cao.warn("no pairs above --kin-min; the PCA is the standard one");
}

void KinshipWhitener::apply(Mat2D& X, const std::vector<Mat2D>& B) const {
#pragma omp parallel for schedule(dynamic)
  for (size_t f = 0; f < families.size(); ++f) {
    const Int1D& idx = families[f];
    Mat2D rows = X(idx, Eigen::all);
    X(idx, Eigen::all) = B[f] * rows;
  }
}
