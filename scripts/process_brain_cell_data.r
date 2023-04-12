library("edgeR")
library("matrixStats")
library(stringr)
library(DescTools)
memory.limit(60000)
setwd("~/BenchmarkingFederated/Dataset2")

# Loads in data
intron_matrix <- read.csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_intron-matrix.csv")
exon_matrix <- read.csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_exon-matrix.csv")
full_matrix <- intron_matrix + exon_matrix


full_matrix_log2cpm <- log2(cpm(full_matrix)+1)

# Replace first row of gene id's
full_matrix[,1] <- intron_matrix[,1]
full_matrix_log2cpm[,1] <- intron_matrix[,1]

# ------------------------------- External CSVs ------------------------------- #

# Get cell labels from external csv
# NOTE: This code was for only the highest cell subtype
# labels <- read.csv("final_two_sample_columns.csv")
# cell.types <- c("Exc", "Inh", "Oli", "Ast", "OPC", "End", "Mic")

# Create cluster columns
samples <- read.csv("human_MTG_gene_expression_matrices_2018-06-14/human_MTG_2018-06-14_samples-columns.csv", stringsAsFactors = FALSE)
labels <- samples[c("sample_name", "cluster")]
labels <- labels[labels$cluster != "no class",]
labels <- cbind(labels, str_split_fixed(labels$cluster, " ", n=Inf))
colnames(labels) <- c("sample_name", "cluster", "higher", "layer", "intermediate", "granular")

cell.types <- unique(labels$cluster)

# Gets list of column indices for a particular cell type to subset
for(type in cell.types) {
  t(assign(paste0(type, ".celltypes"), labels[which(labels$clu == type),1]))
}

# Read in gene.rows file
gene.rows <- read.csv("human_MTG_2018-06-14_genes-rows.csv")

# Replace row names of full matrix with genes
rownames(full_matrix_log2cpm) <- gene.rows[["gene"]]


# ------------------------------- Main Matrix 0's ------------------------------- #

# Remove genes from whole matrix w/ 0 expression
keep <- rowSums(full_matrix_log2cpm[, 2:15929]) > 0
kept.genes <- full_matrix_log2cpm[keep,]


# Subset cell subtypes
for (n in cell.types) {
  temp.cell.types <- get(paste0(n, ".celltypes"))
  temp.subset <- kept.genes[, temp.cell.types]
  assign(paste0(n, ".mm.subset"), temp.subset)
}

# Create empty median common gene matrix
cg.median.mm.df <- data.frame(matrix(ncol = 0, nrow = length(kept.genes[,1])))
rownames(cg.median.mm.df) <- rownames(kept.genes)

# Create empty count common gene matrix
cg.count.mm.df <- data.frame(matrix(ncol = 0, nrow = length(kept.genes[,1])))
rownames(cg.count.mm.df) <- rownames(kept.genes)

# Fill median common gene matrix with row medians from every cell subtype
for (n in cell.types) {
  temp.subset.common <- get(paste0(n, ".mm.subset"))
  temp.subset.median.common <- rowMedians(temp.subset.common)
  cg.median.mm.df <- cbind(cg.median.mm.df, temp.subset.median.common)
  temp.subset.count.common <- rowSums(temp.subset.common > 0)
  cg.count.mm.df <- cbind(cg.count.mm.df, temp.subset.count.common)
}

colnames(cg.median.mm.df) <- cell.types

# Calculate variance of the medians and remove genes w/ 0 variance
cg.median.mm.var <- apply(cg.median.mm.df, 1, var)
keep.var <- which(cg.median.mm.var != 0)
cg.median.mm.var <- cg.median.mm.var[cg.median.mm.var != 0]

# Subset list of genes after removing variance criteria
cg.median.mm.df.var <- cg.median.mm.df[keep.var, ]


# This line of code takes a long time
full_matrix_log2cpm_subset <- rbind(subset(full_matrix_log2cpm, select = -X), samples[["cluster"]])

keep.var.genes <- names(keep.var)
rownames(full_matrix_log2cpm_subset)[50282] <- "Classification"
final.matrix <- full_matrix_log2cpm_subset[c(keep.var.genes, "Classification"),]

write.csv(final.matrix, file = "everything.csv", row.names = TRUE, col.names = TRUE)

