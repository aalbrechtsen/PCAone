args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: Rscript scripts/plot_projection_with_reference.R <ref_eigvecs2> <projected_eigvecs> <projected_fam> <out_pdf>")
}

ref_path <- args[1]
proj_path <- args[2]
fam_path <- args[3]
out_path <- args[4]

ref <- read.table(
  ref_path,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE,
  check.names = FALSE,
  comment.char = ""
)
proj <- read.table(proj_path, header = FALSE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
fam <- read.table(fam_path, header = FALSE, stringsAsFactors = FALSE)

if (ncol(proj) < 2) {
  stop("projected eigvecs file must contain at least two PCs")
}
if (nrow(fam) < 1) {
  stop("projected fam file is empty")
}

proj_label <- paste(fam[1, 1], fam[1, 2], sep = ":")
populations <- unique(ref$`#FID`)
palette <- grDevices::hcl.colors(length(populations), palette = "Dark 3")
names(palette) <- populations

grDevices::pdf(out_path, width = 8, height = 6)
plot(
  ref$PC1,
  ref$PC2,
  col = palette[ref$`#FID`],
  pch = 16,
  cex = 0.8,
  xlab = "PC1",
  ylab = "PC2",
  main = "Reference PCA With Projected Individual"
)
points(proj[1, 1], proj[1, 2], pch = 8, cex = 2.0, lwd = 2, col = "black")
text(proj[1, 1], proj[1, 2], labels = proj_label, pos = 4, offset = 0.7, cex = 0.9)
legend(
  "topright",
  legend = c(populations, proj_label),
  col = c(unname(palette), "black"),
  pch = c(rep(16, length(populations)), 8),
  pt.cex = c(rep(0.8, length(populations)), 1.6),
  bty = "n",
  cex = 0.8
)
grDevices::dev.off()
