#ifndef PCAONE_FILEPLINK_
#define PCAONE_FILEPLINK_

#include "Data.hpp"
#include "Utils.hpp"

class FileBed : public Data {
 public:
  //
  FileBed(const Param &params_) : Data(params_) {
    cao.print(tick.date(), "start parsing PLINK format");
    std::string fbim = params.filein + ".bim";
    std::string ffam = params.filein + ".fam";
    nsamples = count_lines(ffam);
    nsnps = count_lines(fbim);
    cao.print(tick.date(), "N (# samples):", nsamples, ", M (# SNPs):", nsnps);
    snpmajor = true;
    bed_bytes_per_snp = (nsamples + 3) >> 2;
    std::string fbed = params.filein + ".bed";
    bed_ifstream.open(fbed, std::ios::in | std::ios::binary);
    if (!bed_ifstream.is_open()) cao.error("Cannot open bed file.");
    // check magic number of bed file
    uchar header[3];
    bed_ifstream.read(reinterpret_cast<char *>(&header[0]), 3);
    if ((header[0] != 0x6c) || (header[1] != 0x1b) || (header[2] != 0x01))
      cao.error("Incorrect magic number in plink bed file.");
    if (params.center) {
      centered_geno_lookup = Arr2D::Zero(4, nsnps);
      F = Mat1D::Zero(nsnps);
    }
    if (params.project > 0) {
      cao.print(tick.date(), "match target SNPs to the extended bim (.mbim)");
      centered_geno_lookup = Arr2D::Zero(4, nsnps);
      F = Mat1D::Zero(nsnps);
      project_ref_indices.assign(nsnps, -1);
      project_flip.assign(nsnps, 0);

      std::ifstream ref_bim(params.filebim), target_bim(fbim);
      if (!ref_bim.is_open()) cao.error("Cannot open the extended bim (.mbim) file.");
      if (!target_bim.is_open()) cao.error("Cannot open the target bim file.");

      struct RefSite {
        int index = -1;
        std::string chrom, pos;
        std::string a1, a2;
        double freq = 0.0;
      };

      std::unordered_map<std::string, RefSite> ref_sites;
      std::string line;
      int ref_idx = 0;
      while (getline(ref_bim, line)) {
        auto tokens = split_string(line, "\t");
        if ((int)tokens.size() != 7) cao.error("the input file is not valid!\n => " + params.filebim);
        ref_sites.emplace(tokens[1], RefSite{ref_idx++, tokens[0], tokens[3], tokens[4], tokens[5], std::stod(tokens[6])});
      }

      int target_idx = 0;
      while (getline(target_bim, line)) {
        auto tokens = split_string(line, "\t ");
        if ((int)tokens.size() < 6) cao.error("the input file is not valid!\n => " + fbim);
        auto it = ref_sites.find(tokens[1]);
        if (it == ref_sites.end()) {
          project_skipped++;
          target_idx++;
          continue;
        }

        const auto &ref_site = it->second;
        if (tokens[0] != ref_site.chrom || tokens[3] != ref_site.pos) {
          project_skipped++;
          target_idx++;
          continue;
        }

        if (tokens[4] == ref_site.a1 && tokens[5] == ref_site.a2) {
          F(target_idx) = ref_site.freq;
          project_ref_indices[target_idx] = ref_site.index;
          project_overlap++;
        } else if (tokens[4] == ref_site.a2 && tokens[5] == ref_site.a1) {
          F(target_idx) = 1.0 - ref_site.freq;
          project_ref_indices[target_idx] = ref_site.index;
          project_flip[target_idx] = 1;
          project_overlap++;
          project_flipped++;
        } else {
          project_skipped++;
        }
        target_idx++;
      }

      if (project_overlap == 0) cao.error("no overlapping SNPs found between target .bim and reference .mbim");
      cao.print(tick.date(), "projection overlap =", project_overlap, ", flipped =", project_flipped,
                ", skipped =", project_skipped);
    }
  }

  ~FileBed() override = default;

  void read_all() final;
  // for blockwise
  void check_file_offset_first_var() final;

  void read_block_initial(uint64, uint64, bool) final;

  void read_block_update(uint64, uint64, const Mat2D &, const Mat1D &, const Mat2D &, bool) final;

 private:
  std::ifstream bed_ifstream;
  uint64 bed_bytes_per_snp;
  bool frequency_was_estimated = false;
  std::vector<uchar> inbed;
};

PermMat permute_plink(std::string &fin, const std::string &fout, uint gb, uint nbands);

#endif  // PCAONE_FILEPLINK_
