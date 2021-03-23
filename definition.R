############################################################################
#                                                                          #
# Author : DUQUESNOY Marc                                                  #
# Date : 28/09/20                                                          #
# Topic : Definition of heterogeneous electrodes.                          #
#                                                                          #
############################################################################


### Libraries
library(Rcmdr)
library(FactoMineR)
library(car)


### Dataset
Dataset <- read.table("./data/experimental-dataset.csv", header = TRUE,
                      sep = ";", na.strings = "NA", dec = ".",
                      strip.white = TRUE)

### Principal Component Analysis
### The associated code corresponds to the seconde implemented PCA (refer to the manuscript for more details).
res <- PCA(Dataset[,c(3,4,5,6,7,8,9,10,11,12,13)], scale.unit = TRUE, ncp = 5, graph = FALSE)
res.hcpc <- HCPC(res, nb.clust = -1, consol = TRUE, min = 3, max = 10, graph = TRUE)
plot.PCA(res, axes = c(1, 2), choix = "ind", habillage = "none",
         col.ind.sup = "blue", col.quali = "magenta", label = c("ind", "ind.sup", "quali"), 
         new.plot = TRUE)
plot.PCA(res, axes = c(1, 2), choix = "var", new.plot = TRUE, col.var = "black", 
         col.quanti.sup = "blue", label = c("var", "quanti.sup"), lim.cos2.var = 0)


### Clustering analysis
### The associated code corresponds to the boxplot comparison. YOu must change the variables in order to compare different degrees of variation.
Z = res.hcpc$data.clust
boxplot(Z$σ..mg.cm2. ~ Z$clust)
boxplot(Z$σ..μm. ~ Z$clust)

### For the statistical comparison, use the function kruskal.test() to analyze clusters and degrees of variation.
kruskal.test(σ..mg.cm2. ~ clust, data = Z)

### Export the resulting Dataset
write.csv(Z, file="path/to/store/heterogeneity/definition.csv", sep=";")



