#ifndef CMD_H_
#define CMD_H_

#include <cstdint>
#include <sstream>

#include "Common.hpp"

using uint = std::uint32_t;
using uint64 = std::uint64_t;

enum class FileType { PLINK, CSV, BEAGLE, BINARY, BGEN, PGEN };

enum class SvdType { IRAM, PCAoneAlg1, PCAoneAlg2, FULL };

class Param {
 public:
  Param(int argc, char** argv);
  ~Param();

  FileType file_t;  // PLINK, CSV, BEAGLE, BGEN
  SvdType svd_t;
  std::string fileU, fileS, fileE, fileV;
  std::string filein;
  std::string fileout = "pcaone";
  double memory = 0;  // 0 for disable
  uint nsamples = 0;
  uint nsnps = 0;
  uint k = 10;
  uint maxp = 20;  // maximum number of power iterations
  uint threads = 12;
  uint bands = 64;
  bool genetic = true;  // bifle,pgen,bgen
  bool dopca = true;    // false for projection, selection, inbreeding
  bool perm = false;    // wheather data is permuted
  // for normalization
  double scaleFactor = 1;
  // for emu iteration
  uint maxiter = 100;
  double alpha = 0.001;
  // can be tol_emu or tol_pcangsd
  double tolem = 1e-5;
  double tolmaf = 1e-6;
  double maf = 0.0;
  // for arnoldi
  uint ncv = 20;  // max(20, 2*k + 1)
  uint imaxiter = 1000;
  double itol = 1e-6;
  // for halko
  uint oversamples = 10;
  double tol = 1e-4;
  uint buffer = 2;
  uint rand = 1;
  // for ld stuff
  bool print_r2 = false;
  // for ld pruning
  std::string filebim;  // the 7-th column can be MAF
  int ld_stats = 0;     // 0: adj; 1: std
  double ld_r2 = 0;
  bool ld = false;       // true if tolld > 0
  uint ld_bp = 1000000;  // base pairs not number of snps
  // for clumping
  std::string clump;
  std::string assoc_colnames;
  double clump_p1 = 0.0001;
  double clump_p2 = 0.01;
  double clump_r2 = 0.5;
  uint clump_bp = 250000;
  // for projection
  int project = 0;
  uint project_bootstrap = 0;
  bool project_bootstrap_save = false;
  // for inbreeding
  int inbreed = 0;
  // for selection. 1: Galinsky et al. 2. pcadapt
  int selection = 0;

  // general
  uint verbose = 1;                       // verbose level. 0: no verbose; 1: output to screen; 2: debug info
  int scale = SCALE_STANDARDIZE_GENETIC;  // default: standardize genetic data by sqrt(ploidy*f*(1-f))
  bool hardcall = false;
  bool groff = false;
  bool printv = false;
  bool missme = false;  // keep track of missing information
  bool noshuffle = false;
  bool emu = false;
  bool pcangsd = false;  // enable pcangsd procedure
  // bool fancyem = false;  // true if emu/pcangsd + svd 2
  bool mev = true;
  bool out_of_core = false;  // otherwise load all matrix into RAM.
  int ploidy = 2;
  int seed = 112;          // seeding
  bool filterSNP = false;  // filter snps
  bool center = true;      // false if G is raw data likelihood or inbred mode
  bool evaladmix = false;  // compute correlation of residuals (evalAdmix)
  int evaladmix_k = 0;     // PCs used by --evaladmix; 0 means all available
  // for kinship-whitened PCA
  std::string filekin;            // kinship of close relatives, pairs or N x N matrix
  double kin_min = 0.08838834765;  // 2^-3.5, KING lower bound of 2nd degree
  // for --robust
  std::string robust;         // auto, aarobust-kin, detect-white, cswhite, frkin (empty: off)
  int robust_small_max = 1000;
  bool robust_fixed_rank = false;
  bool impute_diag = false;  // GRM diagonal imputed from the off-diagonal entries (standard PCA / aarobust-kin)
  double robust_tol = 1e-6;  // operator engine: convergence tolerance of the subspace iterations
  double king_screen = 0.04;  // only pairs with KING kinship above this may be treated as related
  std::string king_search = "auto";  // operator engine candidates: all (KING over all pairs), sketch, or auto
  bool aar_king_only = false;  // aarobust-kin: KING candidates only (no family-axis / evalAdmix screen)
  std::string ea_search = "auto";  // dwg operator engine, evalAdmix screen: all pairs, residual sketch, or auto (sketch above king_sketch_min)
  int king_sketch_min = 5000;         // --king-search and --ea-search auto: sketch above this N (no accuracy loss seen at N = 2000-5000)
  int king_sketch_dim = 2048;         // columns of the count sketch
  int king_neighbours = 5;            // sketch neighbours checked per individual (extended for large families)
  std::string robust_engine = "auto";  // dense, operator (alias iram) or auto (dense up to robust_dense_max samples)
  int robust_dense_max = 5000;
  bool pcp_iram = false;
  bool pcp_edge = false;  // aarobust-kin: do not lower the PCP threshold below the GRM noise edge (L stays low rank)  // aarobust-kin: partial eigen-thresholding by IRAM (Spectra) instead of full eigendecompositions
  std::string robust_pcs = "operator";  // operator engine, whitening modes: final PCs by the operator iteration or winSVD (--svd 2)
  // bool estpi = false; // true if output pi is needed

  std::ostringstream ss;
};

#endif  // CMD_H_
