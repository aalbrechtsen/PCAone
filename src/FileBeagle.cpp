/*******************************************************************************
 * @file        https://github.com/Zilong-Li/PCAone/src/FileBeagle.cpp
 * @author      Zilong Li
 * Copyright (C) 2022-2024. Use of this code is governed by the LICENSE file.
 ******************************************************************************/

#include "FileBeagle.hpp"

using namespace std;

// read all data and estimate F
void FileBeagle::read_all() {
  if (params.project > 0) {
    auto decode_beagle_allele = [](const std::string& allele) {
      if (allele == "0") return std::string("A");
      if (allele == "1") return std::string("C");
      if (allele == "2") return std::string("G");
      if (allele == "3") return std::string("T");
      return allele;
    };

    struct RefSite {
      int index = -1;
      std::string chrom, pos;
      std::string a1, a2;
      double freq = 0.0;
    };

    std::unordered_map<std::string, RefSite> ref_sites;
    std::unordered_map<std::string, RefSite> ref_sites_by_pos;
    std::ifstream ref_bim(params.filebim);
    if (!ref_bim.is_open()) cao.error("Cannot open the extended bim (.mbim) file.");

    std::string line;
    int ref_idx = 0;
    while (getline(ref_bim, line)) {
      auto tokens = split_string(line, "\t");
      if ((int)tokens.size() != 7) cao.error("the input file is not valid!\n => " + params.filebim);
      RefSite site{ref_idx++, tokens[0], tokens[3], tokens[4], tokens[5], std::stod(tokens[6])};
      ref_sites.emplace(tokens[1], site);
      ref_sites_by_pos.emplace(tokens[0] + "_" + tokens[3], site);
    }

      F = Mat1D::Zero(nsnps);
      G = Mat2D::Zero(nsamples, nsnps);
      C = ArrBool::Zero(nsnps * nsamples);
      P = Mat2D::Zero(nsamples * 2, nsnps);
      project_ref_indices.assign(nsnps, -1);
      project_flip.assign(nsnps, 0);

    fp = gzopen(params.filein.c_str(), "r");
    tgets(fp, &buffer, &bufsize);  // header
    if (buffer != original) original = buffer;
    buffer = original;

    const char *delims = "\t \n";
    char* tok;
    for (uint j = 0; j < nsnps; ++j) {
      tgets(fp, &buffer, &bufsize);
      if (buffer != original) original = buffer;
      tok = strtok_r(buffer, delims, &buffer);
      std::string marker(tok ? tok : "");
      tok = strtok_r(NULL, delims, &buffer);
      std::string allele1 = decode_beagle_allele(tok ? tok : "");
      tok = strtok_r(NULL, delims, &buffer);
      std::string allele2 = decode_beagle_allele(tok ? tok : "");

      auto it = ref_sites.find(marker);
      if (it == ref_sites.end()) it = ref_sites_by_pos.find(marker);
      bool matched = false;
      if (it != ref_sites.end()) {
        const auto& ref_site = it->second;
          if (allele1 == ref_site.a1 && allele2 == ref_site.a2) {
            F(j) = ref_site.freq;
            project_ref_indices[j] = ref_site.index;
            project_overlap++;
            matched = true;
        } else if (allele1 == ref_site.a2 && allele2 == ref_site.a1) {
          F(j) = 1.0 - ref_site.freq;
          project_ref_indices[j] = ref_site.index;
          project_flip[j] = 1;
          project_overlap++;
          project_flipped++;
          matched = true;
        }
      }
        if (!matched) project_skipped++;

        for (uint i = 0; i < nsamples; ++i) {
          tok = strtok_r(NULL, delims, &buffer);
          double p11 = tok ? strtod(tok, NULL) : NAN;
          tok = strtok_r(NULL, delims, &buffer);
          double p12 = tok ? strtod(tok, NULL) : NAN;
          tok = strtok_r(NULL, delims, &buffer);
          double p22 = tok ? strtod(tok, NULL) : NAN;

          if (!matched) continue;

          P(2 * i + 0, j) = p11;
          P(2 * i + 1, j) = p12;

          double psum = p11 + p12 + p22;
        if (!std::isfinite(psum) || psum <= 0.0) {
          C[j * nsamples + i] = 1;
          G(i, j) = 0.0;
          continue;
        }
        double dosage_a1 = (2.0 * p11 + p12) / psum;
        G(i, j) = dosage_a1 / 2.0 - F(j);
        C[j * nsamples + i] = 0;
      }
      buffer = original;
    }

    if (project_overlap == 0) cao.error("no overlapping SNPs found between target BEAGLE and reference .mbim");
    cao.print(tick.date(), "projection overlap =", project_overlap, ", flipped =", project_flipped,
              ", skipped =", project_skipped);
    return;
  }

  P = Mat2D::Zero(nsamples * 2, nsnps);
  fp = gzopen(params.filein.c_str(), "r");
  parse_beagle_file(P, fp, nsamples, nsnps);
  if (!params.pca) return;
  cao.print(tick.date(), "begin to estimate allele frequencies");
  F = Mat1D::Constant(nsnps, 0.25);
  emMAF_with_GL(F, P, params.maxiter, params.tolmaf);
  // filter snps and resize G;
  filter_snps_resize_F();
  // resize P, only keep columns matching the indecis in idx;
  // P = P(Eigen::all, idx).eval(); // aliasing issue!!!
  G = Mat2D::Zero(nsamples, nsnps);  // initial E which is G
#pragma omp parallel for
  for (uint j = 0; j < nsnps; j++) {
    double p0, p1, p2;
    uint s = params.keepsnp ? keepSNPs[j] : j;
    for (uint i = 0; i < nsamples; i++) {
      p0 = P(2 * i + 0, s) * (1.0 - F(j)) * (1.0 - F(j));
      p1 = P(2 * i + 1, s) * 2 * F(j) * (1.0 - F(j));
      p2 = (1 - P(2 * i + 0, s) - P(2 * i + 1, s)) * F(j) * F(j);
      G(i, j) = (p1 + 2 * p2) / (p0 + p1 + p2) - 2.0 * F(j);
    }
  }
}

void FileBeagle::check_file_offset_first_var() {
  if (params.verbose) cao.print("reopen beagle file and read head line");
  fp = gzopen(params.filein.c_str(), "r");
  tgets(fp, &buffer, &bufsize);  // parse header line
  if (buffer != original) original = buffer;
}

void FileBeagle::read_block_initial(uint64 start_idx, uint64 stop_idx, bool standardize = false) {
  if (params.pca) cao.error("doesn't support out-of-core PCAngsd algorithm");
  uint actual_block_size = stop_idx - start_idx + 1;
  if (G.cols() < blocksize || (actual_block_size < blocksize)) {
    P = Mat2D::Zero(nsamples * 2, actual_block_size);
  }
  const char *delims = "\t \n";
  char* tok;
  // read all GL data into P
  for (uint j = 0; j < actual_block_size; ++j) {
    tgets(fp, &buffer, &bufsize);  // get a line
    if (buffer != original) original = buffer;
    tok = strtok_r(buffer, delims, &buffer);
    tok = strtok_r(NULL, delims, &buffer);
    tok = strtok_r(NULL, delims, &buffer);
    for (uint i = 0; i < nsamples; i++) {
      tok = strtok_r(NULL, delims, &buffer);
      P(2 * i + 0, j) = strtod(tok, NULL);
      tok = strtok_r(NULL, delims, &buffer);
      P(2 * i + 1, j) = strtod(tok, NULL);
      tok = strtok_r(NULL, delims, &buffer);
    }
    buffer = original;
  }
}
