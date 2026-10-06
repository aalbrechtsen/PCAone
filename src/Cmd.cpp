/*******************************************************************************
 * @file        https://github.com/Zilong-Li/PCAone/src/Cmd.cpp
 * @author      Zilong Li
 * Copyright (C) 2022-2024. Use of this code is governed by the LICENSE file.
 ******************************************************************************/

#include "Cmd.hpp"

#include <iterator>

#include "popl/popl.hpp"

using namespace popl;

Param::Param(int argc, char** argv) {
  // clang-format off
  bool haploid = false;
  std::string copyr{"PCA All In One (v" + (std::string)VERSION + ")        https://github.com/Zilong-Li/PCAone\n" +
                    "(C) 2021-2024 Zilong Li        GNU General Public License v3\n" +
  "\n" +
                    "Usage: 1) use PLINK files as input and apply default window-based RSVD method\n" +
                    "       $ PCAone -b plink \n\n" +
                    "       2) use CSV file as input and apply the Implicitly Restarted Arnoldi Method\n" +
                    "       $ PCAone -c csv.zst -d 0 \n\n" +
                    "       3) compute ancestry adjusted LD matrix and R2\n" +
                    "       $ PCAone -b plink -k 2 -D -o adj \n" +
                    "       $ PCAone -B adj.residuals -f adj.mbim -R --ld-bp 1000" +
  "\n"};
  OptionParser opts(copyr);
  opts.add<Value<std::string>, Attribute::headline>("","PCAone","General options:");
  auto help_opt = opts.add<Switch>("h", "help", "print all options including hidden advanced options");
  opts.add<Value<double>>("m", "memory", "RAM usage in GB unit for out-of-core mode. default is in-core mode", memory, &memory);
  opts.add<Value<uint>>("n", "threads", "the number of threads to be used", threads, &threads);
  opts.add<Value<uint>>("v", "verbose", "verbosity level for logs. Options are\n"
                                        "0: silent, no messages on screen;\n"
                                        "1: concise messages to screen;\n"
                                        "2: more verbose information;\n"
                                        "3: enable debug information."
                        , verbose, &verbose);
  opts.add<Value<std::string>, Attribute::headline>("","PCA","PCA algorithms:");
  auto svd_opt = opts.add<Value<uint>>("d", "svd", "SVD method to be applied. default 2 is recommended for big data. Options are\n"
                                                   "0: the Implicitly Restarted Arnoldi Method (IRAM);\n"
                                                   "1: the Yu's single-pass Randomized SVD with power iterations;\n"
                                                   "2: the accurate window-based Randomized SVD method (PCAone);\n"
                                                   "3: the full Singular Value Decomposition.", 2);
  opts.add<Value<uint>>("k", "pc", "top k principal components (PCs) to be calculated", k, &k);
  opts.add<Value<int>>("C", "scale", "do normalization or scaling for input file. Options are\n"
                                     "-9: standardize genetic data by sqrt(ploidy*f*(1-f));\n"
                                     " 0: do nothing and proceed to SVD;\n"
                                     " 1: do direct standardization, as the scale(x, center=TRUE, scale=TRUE) function in R;\n"
                                     " 2: do first count per median log transformation (CPMED), then standardization;\n"
                                     " 3: do first log1p transformation, then standardization;\n"
                                     " 4: do first relative counts, then standardization.", scale,  &scale);
  opts.add<Value<uint>>("", "maxp", "maximum number of power iterations for RSVD algorithm.", maxp, &maxp);
  opts.add<Switch>("S", "no-shuffle", "do not shuffle columns of data for --svd 2 (if not locally correlated).", &noshuffle);
  opts.add<Value<uint>, Attribute::advanced>("w", "batches", "the number of mini-batches used by --svd 2.", bands, &bands);
  opts.add<Value<int>>("", "seed", "seeds for reproducing results.\n", seed, &seed);
  opts.add<Switch>("", "emu", "use EMU algorithm for genotype input with missingness.", &emu);
  opts.add<Switch>("", "pcangsd", "use PCAngsd algorithm for genotype likelihood input.", &pcangsd);
  opts.add<Value<uint>, Attribute::advanced>("", "M", "the number of features (eg. SNPs) if already known.", 0, &nsnps);
  opts.add<Value<uint>, Attribute::advanced>("", "N", "the number of samples if already known.", 0, &nsamples);
  opts.add<Value<double>, Attribute::advanced>("", "scale-factor", "feature counts for each sample are normalized and multiplied by this value", 1.0, &scaleFactor);
  opts.add<Value<uint>, Attribute::advanced>("", "buffer", "memory buffer in GB unit for permuting the data.", buffer, &buffer);
  opts.add<Value<uint>, Attribute::advanced>("", "imaxiter", "maximum number of IRAM iterations.", imaxiter, &imaxiter);
  opts.add<Value<double>, Attribute::advanced>("", "itol", "stopping tolerance for IRAM algorithm.", itol, &itol);
  opts.add<Value<uint>, Attribute::advanced>("", "ncv", "the number of Lanzcos basis vectors for IRAM.", ncv, &ncv);
  opts.add<Value<uint>, Attribute::advanced>("", "oversamples", "the number of oversampling columns for RSVD.", oversamples, &oversamples);
  opts.add<Value<uint>, Attribute::advanced>("", "rand", "the random matrix type. 0: uniform; 1: guassian.", rand, &rand);
  opts.add<Value<uint>, Attribute::advanced>("", "maxiter", "maximum number of EM iterations.", maxiter, &maxiter);
  opts.add<Value<double>, Attribute::advanced>("", "tol-rsvd", "tolerance for RSVD algorithm.", tol, &tol);
  opts.add<Value<double>, Attribute::advanced>("", "tol-em", "tolerance for EMU/PCAngsd algorithm.", tolem, &tolem);
  opts.add<Value<double>, Attribute::advanced>("", "tol-maf", "tolerance for MAF estimation by EM.", tolmaf, &tolmaf);
  
  opts.add<Value<std::string>, Attribute::headline>("","INPUT","Input options:");
  auto plinkfile = opts.add<Value<std::string>>("b", "bfile", "prefix of PLINK .bed/.bim/.fam files.", "", &filein);
  opts.add<Switch, Attribute::advanced>("", "haploid", "the plink format represents haploid data.", &haploid);
  auto pgenfile = opts.add<Value<std::string>>("p", "pgen", "prefix of PLINK2 .pgen/.pvar/.psam files.", "", &filein);
  opts.add<Switch, Attribute::advanced>("", "hardcall", "use hardcall genotype instead of dosages.", &hardcall);
  auto binfile = opts.add<Value<std::string>>("B", "binary", "path of binary file.", "", &filein);
  auto csvfile = opts.add<Value<std::string>>("c", "csv", "path of comma seperated CSV file compressed by zstd.", "", &filein);
  auto bgenfile = opts.add<Value<std::string>>("g", "bgen", "path of BGEN file compressed by gzip/zstd.", "", &filein);
  auto beaglefile = opts.add<Value<std::string>>("G", "beagle", "path of BEAGLE file compressed by gzip.", "", &filein);
  opts.add<Value<std::string>>("F", "match-bim", "the .mbim file to be matched, where the 7th column is allele frequency.", "", &filebim);
  auto usvprefix = opts.add<Value<std::string>>("P", "USV", "prefix of PCAone .eigvecs/.sigvals/.loadings/.mbim.");
  opts.add<Value<std::string>, Attribute::hidden>("", "read-U", "path of file with left singular vectors (.eigvecs).", "", &fileU);
  opts.add<Value<std::string>, Attribute::hidden>("", "read-V", "path of file with right singular vectors (.loadings).", "", &fileV);
  opts.add<Value<std::string>, Attribute::hidden>("", "read-S", "path of file with sigular values (.sigvals).", "", &fileS);
  
  opts.add<Value<std::string>, Attribute::headline>("","OUTPUT","Output options:");
  opts.add<Value<std::string>>("o", "out", "prefix of output files. default [pcaone].", fileout, &fileout);
  opts.add<Switch>("V", "printv", "output the right eigenvectors with suffix .loadings.", &printv);
  opts.add<Switch>("D", "ld", "output a binary matrix for downstream LD related analysis.", &ld);
  opts.add<Switch>("R", "print-r2", "print LD R2 to *.ld.gz file for pairwise SNPs within a window controlled by --ld-bp.", &print_r2);
  
  opts.add<Value<std::string>, Attribute::headline>("","MISC","Misc options:");
  opts.add<Value<double>>("", "maf", "exclude variants with MAF lower than this value", maf, &maf);
  opts.add<Value<int>>("", "project", "project the new samples onto the existing PCs. Options are\n"
                                      "0: disabled;\n"
                                      "1: by multiplying the loadings with mean imputation for missing genotypes;\n"
                                      "2: by solving the least squares system Vx=g. skip sites with missingness;\n"
                                      "3: by EM to account for genotype uncertainty (BEAGLE input);\n"
                                      "4: by Augmentation, Decomposition and Procrusters transformation.\n", project, &project);
  opts.add<Value<uint>>("", "project-bootstrap", "run SNP bootstrap diagnostics for --project 2 using this many replicates.", project_bootstrap, &project_bootstrap);
  opts.add<Switch>("", "project-bootstrap-save", "save raw bootstrap projection coordinates to *.proj.bootstrap.eigvecs.", &project_bootstrap_save);
  opts.add<Value<int>>("", "inbreed", "compute the inbreeding coefficient accounting for population structure. Options are\n"
                                      "0: disabled;\n"
                                      "1: compute per-site inbreeding coefficient and HWE test.\n", inbreed, &inbreed);
  opts.add<Switch>("", "evaladmix", "compute the correlation of residuals (evalAdmix) given the top PCs.", &evaladmix);
  opts.add<Value<int>>("", "evaladmix-k", "number of PCs used by --evaladmix. default is all computed PCs (use K-1 for an admixture model with K populations).", evaladmix_k, &evaladmix_k);
  opts.add<Value<std::string>>("", "kinship", "kinship of close relatives for a PCA not driven by families (kinship-whitened PCA).\n"
                                              "either a pair table with columns ID1 ID2 KINSHIP (e.g. KING/PLINK2 .kin0, pcaone-ibd .ibd)\n"
                                              "or an N x N matrix (e.g. the .kinship of --evaladmix).", "", &filekin);
  opts.add<Value<double>>("", "kin-min", "pairs with kinship below this are treated as unrelated by --kinship (default 2^-3.5, i.e. 2nd degree).", kin_min, &kin_min);
  auto robust_opt = opts.add<Implicit<std::string>>("", "robust", "PCA robust to close relatives (2nd degree and closer); --robust alone = auto. Modes are\n"
                                             "auto: dwg; with --impute-diag aarobust-kin;\n"
                                             "aarobust-kin: robust PCA of the GRM, diagonal unobserved, kinship threshold (does not depend on -k);\n"
                                             "detect-white: detect related pairs on the raw Gram matrix (diagonal free), then whitening;\n"
                                             "cswhite: CS whitening with KING (or --kinship) kinship; assumes HWE;\n"
                                             "frkin: fixed rank + kinship threshold on the raw Gram matrix (diagonal free);\n"
                                             "dwg: detect-white on the GRM scale, kinship from the fitted noise (no HWE), family axes and an admixture-aware evalAdmix + k0 screen.", "auto");
  opts.add<Switch>("", "impute-diag", "PCA of the GRM with its diagonal imputed from the off-diagonal entries (small N: the observed\n"
                                       "diagonal reflects heterozygosity, not structure). Alone: standard PCA with the diagonal imputed;\n"
                                       "with --robust aarobust-kin: PCs of the fitted structure L instead of the GRM without the related pairs.", &impute_diag);
  opts.add<Value<std::string>>("", "robust-engine", "--robust: dense (N x N in memory), operator (matrix-free block iteration, large N, needs --kinship candidates) or auto (dense up to --robust-dense-max samples).", robust_engine, &robust_engine);
  opts.add<Switch>("", "robust-fixed-rank", "--robust detect-white: use rank k + 1 for the detection fit instead of choosing it from the noise edge (at most k + 1).", &robust_fixed_rank);
  opts.add<Value<double>, Attribute::advanced>("", "robust-tol", "--robust-engine operator: convergence tolerance (relative residual of the structure eigenpairs and change of the fit).", robust_tol, &robust_tol);
  opts.add<Value<int>, Attribute::advanced>("", "robust-small-max", "--robust auto --impute-diag: largest N that uses aarobust-kin.", robust_small_max, &robust_small_max);
  opts.add<Value<std::string>, Attribute::advanced>("", "robust-pcs", "--robust-engine operator, detect-white and cswhite: compute the final PCs of the whitened genotypes with the operator iteration (operator) or the window-based RSVD of --svd 2 (winsvd).", robust_pcs, &robust_pcs);
  opts.add<Switch, Attribute::advanced>("", "pcp-edge", "--robust aarobust-kin / --impute-diag: stop lowering the PCP threshold at the noise edge of the GRM, so the fit stays low rank.", &pcp_edge);
  opts.add<Switch, Attribute::advanced>("", "pcp-iram", "--robust aarobust-kin / --impute-diag: compute only the eigenpairs above the PCP threshold with IRAM instead of full eigendecompositions.", &pcp_iram);
  opts.add<Value<int>, Attribute::advanced>("", "robust-dense-max", "--robust-engine auto: largest N for the dense engine.", robust_dense_max, &robust_dense_max);
  opts.add<Value<std::string>>("", "king-search", "--robust-engine operator without --kinship: how candidate pairs are found. all: KING-robust over all pairs; sketch: KING-robust for each individual's nearest neighbours in a genotype sketch (very large N); auto: sketch above --king-sketch-min samples.", king_search, &king_search);
  opts.add<Value<int>, Attribute::advanced>("", "king-sketch-min", "--king-search auto: smallest N + 1 that uses the sketch.", king_sketch_min, &king_sketch_min);
  opts.add<Value<int>, Attribute::advanced>("", "king-sketch-dim", "--king-search sketch: number of sketch columns.", king_sketch_dim, &king_sketch_dim);
  opts.add<Value<int>, Attribute::advanced>("", "king-neighbours", "--king-search sketch: neighbours checked per individual (doubled while all of them are relatives).", king_neighbours, &king_neighbours);
  opts.add<Value<double>>("", "king-screen", "--robust: only pairs with KING-robust kinship above this may be treated as related.", king_screen, &king_screen);
  opts.add<Value<int>>("", "selection", "compute selection statistics. Options are\n"
                                      "0: disabled;\n"
                                      "1: perform selection scan using Galinsky et al method;\n"
                                      "2: perform selection scan using PCAdapt method.\n", selection, &selection);
  opts.add<Value<double>>("", "ld-r2", "R2 cutoff for LD-based pruning (usually 0.2).", ld_r2, &ld_r2);
  opts.add<Value<uint>>("", "ld-bp", "physical distance threshold in bases for LD window.", ld_bp, &ld_bp);
  opts.add<Value<int>>("", "ld-stats", "statistics to compute LD R2 for pairwise SNPs. Options are\n"
                                       "0: the ancestry adjusted, i.e. correlation between residuals;\n"
                                       "1: the standard, i.e. correlation between two alleles.\n", ld_stats, &ld_stats);
  auto clumpfile = opts.add<Value<std::string>>("", "clump", "assoc-like file with target variants and pvalues for clumping.", "", &clump);
  auto assocnames = opts.add<Value<std::string>>("", "clump-names", "column names in assoc-like file for locating chr, pos and pvalue.", "CHR,BP,P", &assoc_colnames);
  opts.add<Value<double>>("", "clump-p1", "significance threshold for index SNPs.", clump_p1, &clump_p1);
  opts.add<Value<double>>("", "clump-p2", "secondary significance threshold for clumped SNPs.", clump_p2, &clump_p2);
  opts.add<Value<double>>("", "clump-r2", "r2 cutoff for LD-based clumping.", clump_r2, &clump_r2);
  opts.add<Value<uint>>("", "clump-bp", "physical distance threshold in bases for clumping.", clump_bp, &clump_bp);
  opts.add<Switch, Attribute::hidden>("", "groff", "PCAone 1 \"24 December 2024\" \"PCAone-v"+ std::string(VERSION)+"\"  \"Bioinformatics tools\"", &groff);
  
  // collect command line options acutal in effect
  ss << (std::string) "PCAone (v" + VERSION + ")    https://github.com/Zilong-Li/PCAone\n";
  ss << "Options in effect:\n";
  std::copy(argv, argv + argc, std::ostream_iterator<char *>(ss, " "));
  // clang-format on
  try {
    // --robust takes an optional mode: accept "--robust <mode>" as well as "--robust=<mode>"
    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 0; i + 1 < args.size(); ++i)
      if (args[i] == "--robust" && !args[i + 1].empty() && args[i + 1][0] != '-') {
        args[i] += "=" + args[i + 1];
        args.erase(args.begin() + i + 1);
      }
    std::vector<char*> argv2;
    for (auto& a : args) argv2.push_back(a.data());
    opts.parse((int)argv2.size(), argv2.data());
    if (robust_opt->is_set()) robust = robust_opt->value();
    if (groff) {
      GroffOptionPrinter groff_printer(&opts);
      std::cout << groff_printer.print(Attribute::advanced);
      exit(EXIT_SUCCESS);
    }
    if (!opts.unknown_options().empty()) {
      for (const auto& uo : opts.unknown_options()) std::cerr << "unknown option: " << uo << "\n";
      exit(EXIT_FAILURE);
    }
    if (svd_opt->value() == 0)
      svd_t = SvdType::IRAM;
    else if (svd_opt->value() == 1)
      svd_t = SvdType::PCAoneAlg1;
    else if (svd_opt->value() == 2)
      svd_t = SvdType::PCAoneAlg2;
    else if (svd_opt->value() == 3)
      svd_t = SvdType::FULL;
    else
      svd_t = SvdType::PCAoneAlg2;

    if (plinkfile->is_set())
      file_t = FileType::PLINK;
    else if (binfile->is_set())
      file_t = FileType::BINARY;
    else if (bgenfile->is_set())
      file_t = FileType::BGEN;
    else if (beaglefile->is_set())
      file_t = FileType::BEAGLE;
    else if (csvfile->is_set())
      file_t = FileType::CSV;
    else if (pgenfile->is_set())
      file_t = FileType::PGEN;
    else if (help_opt->count() == 1) {
      std::cout << opts.help(Attribute::advanced) << "\n";
      exit(EXIT_SUCCESS);
    } else if (argc == 1) {
      std::cout << opts << "\n";
      exit(EXIT_SUCCESS);
    }
    genetic = (file_t == FileType::PLINK || file_t == FileType::BGEN || file_t == FileType::PGEN);
    // handle PI, i.e U,S,V
    if (usvprefix->is_set()) {
      if (fileU.empty()) fileU = usvprefix->value() + ".eigvecs";
      if (fileE.empty()) fileE = usvprefix->value() + ".eigvals";
      if (fileS.empty()) fileS = usvprefix->value() + ".sigvals";
      if (fileV.empty()) fileV = usvprefix->value() + ".loadings";
      if (filebim.empty()) filebim = usvprefix->value() + ".mbim";
    }

    // handle LD
    if (print_r2 || ld_r2 > 0 || !clump.empty()) {
      dopca = false;  // we always want to center the G for calculating R2
      memory /= 2.0;  // adjust memory estimator
    }

    // handle projection
    if (project > 0) {
      if (project < 1 || project > 3) throw std::invalid_argument("--project supports only 1, 2, or 3 in this release");
      if (fileV.empty() || fileS.empty()) throw std::invalid_argument("please use --USV together with --project");
      if (project_bootstrap > 0 && project != 2)
        throw std::invalid_argument("--project-bootstrap currently supports only --project 2");
      dopca = false, missme = true, out_of_core = false;
      memory = 0;
    } else if (project_bootstrap > 0 || project_bootstrap_save) {
      throw std::invalid_argument("--project-bootstrap requires --project 2");
    }

    // handle selection
    if (selection > 0) {
      if (file_t != FileType::PLINK && file_t != FileType::PGEN)
        throw std::invalid_argument("only supports --bfile/--pgen for now");
      if (fileU.empty() || fileE.empty()) throw std::invalid_argument("please use --USV together with --selection");
      dopca = true;  // we need this to init F
    }

    // handle inbreeding
    if (inbreed > 0) {
      dopca = false, center = false;
      if (fileU.empty() || fileV.empty() || fileS.empty())
        throw std::invalid_argument("please use --USV together with --inbreed");
    }

    // handle memory and misc options
    ncv = 20 > (2 * k + 1) ? 20 : (2 * k + 1);
    oversamples = oversamples > k ? oversamples : k;
    if (haploid && genetic) ploidy = 1;
    if (memory > 0 && svd_t != SvdType::FULL) out_of_core = true;

    filterSNP = maf > 0 ? true : false;  // filter SNP if MAf applied
    if (filterSNP) {
      if (!(maf > 0 && maf < 0.5)) throw std::invalid_argument("--maf has to be between (0, 0.5)");
      if (out_of_core) throw std::invalid_argument("does not support --maf filters for out-of-core mode yet! ");
    }

    // handle EM-PCA
    if (dopca && file_t == FileType::BEAGLE && robust.empty()) pcangsd = true;
    if (emu || pcangsd) {
      missme = true;
    } else if (dopca) {
      maxiter = 0;
    }
    if (out_of_core && pcangsd && (file_t == FileType::BEAGLE))
      throw std::invalid_argument("not supporting -m option (out-of-core) for PCAngsd and BEAGLE input yet!");
    if (!robust.empty()) {
      if (robust != "auto" && robust != "detect-white" && robust != "cswhite" && robust != "frkin" && robust != "aarobust-kin" && robust != "dwg")
        throw std::invalid_argument("--robust must be one of auto, aarobust-kin, detect-white, cswhite, frkin, dwg");
      if (file_t != FileType::PLINK && file_t != FileType::PGEN && file_t != FileType::BEAGLE)
        throw std::invalid_argument("--robust supports --bfile, --pgen and BEAGLE (-G) input");
      if (file_t == FileType::BEAGLE) {
        if (robust != "auto" && robust != "dwg")
          throw std::invalid_argument("--robust with genotype likelihoods (-G) supports dwg (or auto) only");
        if (out_of_core) throw std::invalid_argument("--robust with genotype likelihoods (-G) runs in-core only");
        if (impute_diag) throw std::invalid_argument("--impute-diag does not support genotype likelihoods (-G)");
      }
      if (emu || pcangsd || project > 0 || selection > 0 || inbreed > 0 || ld || evaladmix)
        throw std::invalid_argument("--robust can not be combined with --emu, --pcangsd, --project, --selection, --inbreed, -D or --evaladmix");
      if (robust_engine != "dense" && robust_engine != "operator" && robust_engine != "iram" && robust_engine != "auto")
        throw std::invalid_argument("--robust-engine must be dense, operator or auto");
      if (k < 1) throw std::invalid_argument("--robust needs -k >= 1");
      if (robust_pcs != "operator" && robust_pcs != "winsvd")
        throw std::invalid_argument("--robust-pcs must be operator or winsvd");
      if (king_search != "auto" && king_search != "all" && king_search != "sketch")
        throw std::invalid_argument("--king-search must be all, sketch or auto");
      if (king_sketch_dim < 64 || king_neighbours < 1)
        throw std::invalid_argument("--king-sketch-dim must be >= 64 and --king-neighbours >= 1");
      if (impute_diag && robust != "auto" && robust != "aarobust-kin")
        throw std::invalid_argument("--impute-diag works with standard PCA and --robust aarobust-kin; the Chen & Storey "
                                    "based modes correct the diagonal already");
    }
    if (impute_diag && robust.empty()) {
      // standard PCA with the GRM diagonal imputed: run through the dense --robust path
      if (file_t != FileType::PLINK && file_t != FileType::PGEN)
        throw std::invalid_argument("--impute-diag supports --bfile/--pgen input only");
      if (emu || pcangsd || project > 0 || selection > 0 || inbreed > 0 || ld || evaladmix || !filekin.empty())
        throw std::invalid_argument("--impute-diag can not be combined with --emu, --pcangsd, --project, --selection, --inbreed, -D, --evaladmix or --kinship");
      robust = "diag-impute";
    }
    if (!filekin.empty() && robust.empty()) {
      if (!dopca || project > 0 || selection > 0 || inbreed > 0)
        throw std::invalid_argument("--kinship only works for computing PCs");
      if (emu || pcangsd || file_t == FileType::BEAGLE)
        throw std::invalid_argument("--kinship does not support --emu, --pcangsd or BEAGLE input yet");
      if (ld) throw std::invalid_argument("--kinship can not be used with -D/--ld");
      if (!(kin_min > 0 && kin_min < 0.5)) throw std::invalid_argument("--kin-min has to be between (0, 0.5)");
    }
    if (bands < 4 || bands % 2 != 0)
      throw std::invalid_argument("the -w/--batches must be a power of 2 and the minimun is 4.");

    if (svd_t == SvdType::PCAoneAlg2 && !noshuffle) perm = true;

  } catch (const popl::invalid_option& e) {
    std::cerr << "Invalid Option Exception: " << e.what() << "\n";
    std::cerr << "error:  ";
    if (e.error() == invalid_option::Error::missing_argument)
      std::cerr << "missing_argument\n";
    else if (e.error() == invalid_option::Error::invalid_argument)
      std::cerr << "invalid_argument\n";
    else if (e.error() == invalid_option::Error::too_many_arguments)
      std::cerr << "too_many_arguments\n";
    else if (e.error() == invalid_option::Error::missing_option)
      std::cerr << "missing_option\n";

    if (e.error() == invalid_option::Error::missing_option) {
      std::string option_name(e.option()->name(OptionName::short_name, true));
      if (option_name.empty()) option_name = e.option()->name(OptionName::long_name, true);
      std::cerr << "option: " << option_name << "\n";
    } else {
      std::cerr << "option: " << e.option()->name(e.what_name()) << "\n";
      std::cerr << "value:  " << e.value() << "\n";
    }
    exit(EXIT_FAILURE);
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
    exit(EXIT_FAILURE);
  }
}

Param::~Param() {}
