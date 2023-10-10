
##### Dependencies #####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import pickle
import hdbscan
# The continous entropy sort feature weighting package (cESFW), can be installed from https://github.com/aradley/cESFW
import cESFW

##### Dependencies #####

##### UMAP Plotting Function #####

def Plot_UMAP(Embedding,Sample_Info):
    fig, axs = plt.subplots(1,2,figsize=(16, 8))
    plt.subplot(1,2,1)
    plt.title("Timepoints",fontsize=18)
    Timepoints = np.asarray(Sample_Info["EmbryoDay"]).astype("f")
    Unique_Timepoints = np.unique(Timepoints)
    Unique_Timepoints = np.delete(Unique_Timepoints,np.where(np.isnan(Unique_Timepoints)))
    for i in np.arange(Unique_Timepoints.shape[0]):
        IDs = np.where(Timepoints == Unique_Timepoints[i])[0]
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=8,label=Unique_Timepoints[i])
    #
    plt.legend()
    #
    plt.subplot(1,2,2)
    plt.title("Datasets",fontsize=18)
    Datasets = np.asarray(Sample_Info["Dataset"])
    Unique_Datasets = np.unique(Datasets)
    for i in np.arange(Unique_Datasets.shape[0]):
        IDs = np.where(Datasets == Unique_Datasets[i])[0]
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=8,label=Unique_Datasets[i])
    #
    plt.legend()
    #
    plt.setp(axs, xlabel='UMAP 1', ylabel="UMAP 2")

##### Dependencies #####


## Set path for retreiving and depositing data
human_embryo_path = "/mnt/c/Users/arthu/OneDrive - University of Exeter/Documents/cESFW_Paper/cESFW_Human_Embryo_Topology/Data/Human_Embryo_Analysis/"

## Load human pre and post implantation embryo sample info and scRNA-seq counts matrix.
Human_Sample_Info = pd.read_csv(human_embryo_path+"Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv(human_embryo_path+"Human_Embryo_Counts.csv",header=0,index_col=0)

Drop_Mitochondrial_Genes = Human_Embryo_Counts.columns[np.where(Human_Embryo_Counts.columns.str.contains("MTRNR"))[0]]
Human_Embryo_Counts = Human_Embryo_Counts.drop(Drop_Mitochondrial_Genes,axis=1)

##### Data pre-processing #####

### Feature normalisation ###

## Prior to using cESFW, data must be scaled/normalised such that every feature only has values between 0 and 1.
# How this is done is ultimitely up to the user. However, for scRNA-seq data, we tend to find that the following relitively simple
# normalisation approach yeilds good results.

## Note, cESFW takes each row to be a sample and each column to be a feature. Hence, in this example, each row of Human_Embryo_Counts
# is a cell and each colum is gene.

## Create the scaled matrix from the scRNA-seq counts matrix
Scaled_Matrix = Human_Embryo_Counts.copy()

## Optional: Log transform the data. Emperically we find that in most cases, log transformation of the data
# appears to lead to poorer results further downstream in most cases. However, in some datasets
# we have worked with, it has lead to improved results. This is obviously dependent on what downstream analysis
# the user chooses to do and how they do it, but we recommend starting without any log transformation (hence the
# next line of code being commented out).
#### Scaled_Matrix = np.log2(Scaled_Matrix+1) ####

## Clip the top 2.5 percent of observed values for each gene to mitigate the effect of unusually high
# counts observations.
Upper = np.percentile(Scaled_Matrix,97.5,axis=0)
Scaled_Matrix = Scaled_Matrix.clip(upper=Upper,axis=1) 

## Normalise each feature/gene of the clipped matrix.
Normalisation_Values = np.max(Scaled_Matrix,axis=0)
Scaled_Matrix = Scaled_Matrix / Normalisation_Values

### Run cESFW ###

## Given the scaled matrix, cESFW will use the following function to extract all the non-zero values into a single vector. We do this
# because ES calculations can completely ignore 0 values in the data. For sparse data like scRNA-seq data, this dramatically reduces the memory
# required, and the number of calculations that need to be carried out. For relitively dense data, this step will still need to be carried
# out to use cESFW, but will provide little benifit computationally.

## path: A string path pre-designated folder to deposit the computationally efficient objects. E.g. "/mnt/c/Users/arthu/Test_Folder/"
## Scaled_Matrix: The high dimensional DataFrame whose features have been scaled to values between 0 and 1. Format must be a Pandas DataFrame.
## Min_Minority_State_Cardinality: The minimum value of the total minority state mass that a feature contains before it will be automatically
# removed from the data, and hence analysis.

cESFW.Create_ESFW_Objects(human_embryo_path, Scaled_Matrix, Min_Minority_State_Cardinality = 20)

## Now that we have the compute efficient object, we can calculate the ESSs and EPs matricies. The ESSs matrix provides the pairwise 
# Entropy Sort Scores for each gene in the data. THe EPs matrix provides the EPs pairwise for each gene.

#Masked_ESSs = cESFW.Parallel_Calculate_ESS_EPs(human_embryo_path)
ESSs, EPs = cESFW.Parallel_Calculate_ESS_EPs(human_embryo_path)

## Un-comment below lines to save results for future useage.
# np.save(human_embryo_path + "ESSs.npy",ESSs)
# np.save(human_embryo_path + "EPs.npy",EPs)

### Simple dataset/batch correction by feature exclusion ###

## An attractive property of the cESFW workflow that we present here is that we can obtain a high resolution UMAP
# embedding without having to augment the original counts matrix through smoothing or feature extraction. Data augmentation
# is typical in many scRNA-seq pipelines, and imputation through smoothing/regression, or feature extraction through methods such as PCA,
# are applied in an attempt to mitigate the contribution of noisy signals or batch effects. However, whenever a smoothign or transformation
# is applied to complex high dimensional data, there is a substantial risk of removing feature correlations/strcuture that is informative
# of the biological process of interest. 
#
## cESFW allows us to avoid the use of smoothing/feature extraction methods by simply identifying a set of genes that are
# significantly structured with one another throughout the data. Perhaps more importnatly, this allows us to ignore the thousands
# of features/genes that are noisy and uninformative of teh biological process, which when left in the data for downstream analysis,
# obsfucate the biological signal of interest. 
# 
# However, genes that are consitently up/down-regualated in a specific batch or dataset due to experimental variation or bias, can
# also be constituted as "highly structured" with one another, and hence should be highlighted by cESFW as informative. Hence, we will
# pre-filter out genes that are significantly and specifically present/absent in a single dataset, so that they cannot manifest as batch
# effects in downstream analysis/visualisation. The assumption behind this pre-filtering is that if a gene is significantly localised to one
# dataset, it is likely a counfounding batch effect. This assumption is reliant on each individual dataset consiting of multiple distinct
# cell types. If a dataset consists almost entirely of one cell type that is not present in any of the other datasets, then biologically
# meaningful gene signatures for that cell type will also look like batch effects and be arbitraility removed.
#
## Crucially, this simple feature exclusion methodology does not require any data augmentation to acheive dataset integration.
# As such, we can be more confident that any gene expression patterns we identify downstream are real signals, rather than computational
# artefacts.

## Load the feature IDs that were saved when creating the computational efficient objects.
Used_Features = np.load(human_embryo_path + "Used_Features.npy",allow_pickle=True)
ESSs = np.load(human_embryo_path + "ESSs.npy")
EPs = np.load(human_embryo_path + "EPs.npy")

Dataset_IDs = np.unique(Human_Sample_Info["Dataset"])
Dataset_ESSs = np.zeros((Used_Features.shape[0],Dataset_IDs.shape[0]))
Dataset_EPs = np.zeros((Used_Features.shape[0],Dataset_IDs.shape[0]))

for i in np.arange(Dataset_IDs.shape[0]):
    Dataset_Labels = np.zeros(Human_Sample_Info.shape[0])
    Dataset_Labels[np.where(Human_Sample_Info["Dataset"]==Dataset_IDs[i])[0]] = 1
    # If the number of samples in the dataset is greater than half the cardinality of the entire data, flip the majority/minority state labels.
    if np.sum(Dataset_Labels) > (Human_Sample_Info.shape[0]/2):
        Dataset_Labels = (Dataset_Labels * - 1) + 1
    ## Calulate the ESSs and EPs of the dataset with every feature/gene in the scaled matrix.
    Individual_ESSs, Individual_EPs = cESFW.Calculate_Individual_ESS_EPs(Dataset_Labels,human_embryo_path)
    ## Create a ranked list of features the are enriched in the dataset
    Dataset_ESSs[:,i] = Individual_ESSs
    Dataset_EPs[:,i] = Individual_EPs

Dataset_ESSs = pd.DataFrame(Dataset_ESSs,columns=Dataset_IDs)
Dataset_EPs = pd.DataFrame(Dataset_EPs,columns=Dataset_IDs)

## We will now exclude all genes that have an EP > 0 with any of the datasets from downstream analysis,
# under the assumption that their enrichment in these datasets occours to a degree greater than random chance.
# There are obviously other criteria that one could use which may be better in different cases.

Exclude_Feature_Inds = np.where(Dataset_EPs.max(axis=1) > 0)[0]
Subset_Use_Feature_Inds = np.delete(np.arange(Used_Features.shape[0]),Exclude_Feature_Inds)

## Note which featue names remain.
Subset_Used_Features = Used_Features[Subset_Use_Feature_Inds]

## Subset Masked_ESSs to the remaining features. Masked refers to process of finding all indicies in in EPs matrix that are less than 0
# and setting the values at these indexes to zero in the ESSs or EPs matrix. This provides a mathmatically defined way to turn and dense
# correlation matrix into a sparse matrix where non-significant gene-gene correlations are ignored.
ESSs = ESSs[np.ix_(Subset_Use_Feature_Inds,Subset_Use_Feature_Inds)]
EPs = EPs[np.ix_(Subset_Use_Feature_Inds,Subset_Use_Feature_Inds)]

Masked_ESSs = ESSs.copy()
Masked_ESSs[EPs < 0] = 0
Masked_EPs = EPs.copy()
Masked_EPs[EPs < 0] = 0

### Feature selection importance weighting ###

# Some genes that we know are important in early human embryo development
Known_Important_Genes = np.array(["NANOG","SOX17","GATA4","PDGFRA","PRSS3"])
Known_Important_Gene_Inds = np.where(np.isin(Subset_Used_Features,Known_Important_Genes))[0]

## The ESSs matrix quantifies the correlations/node edges for every gene with every other gene in the data. The EPs provides an value that
# quantifies if the pariwise correlations are more significant than random chance. We would like to focus our analysis on genes that have
# non-random correlations with one another. We can get an unsupervised importance weight for each gene by taking the column averages of the
# ESSs matrix, while weighting the importance of each correlation by the EPs. By using Masked_EPs as the weights, we are stating that 
# anything with an EP < 0 is more likely to have occured by random chance, and hence we do not want to take these relationships into account.
Feature_Weights = np.average(np.absolute(ESSs),weights=Masked_EPs,axis=0)

## Feature_Weights gives an average weight for the significance of each feature. We now add a bioloigcal assumption to workflow analysis 
# that appears to significantly improve gene selection in scRNA-seq data. The assumption is that gene regulatory networks that control 
# cell fate decisions consist of relitively small sets of highly correlated genes. Because Entropy Sorting and the Error Potential implicitly 
# tell us when features are significantly correlated with each other, we can normalise the Feature_Weights of each gene by the number of edges
# that connect them to other genes. This penalises genes that are part of very large networks.

Significant_Genes_Per_Gene = (Masked_EPs > 0).sum(1)
Normalised_Network_Feature_Weights = Feature_Weights/Significant_Genes_Per_Gene

## We can visualise Feature_Weights and Normalised_Network_Feature_Weights in histograms
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.hist(Feature_Weights,bins=30)
xcoords = Feature_Weights[Known_Important_Gene_Inds]
for xc in xcoords:
    plt.axvline(x=xc,c="red")
plt.xlabel("Weight",fontsize=16)
plt.ylabel("Frequency",fontsize=16)
plt.subplot(1,2,2)
plt.title("Normalised_Network_Feature_Weights",fontsize=20)
plt.hist(Normalised_Network_Feature_Weights,bins=30)
xcoords = Normalised_Network_Feature_Weights[Known_Important_Gene_Inds]
for xc in xcoords:
    plt.axvline(x=xc,c="red")
plt.xlabel("Weight",fontsize=16)
plt.ylabel("Frequency",fontsize=16)
# plt.savefig(human_embryo_path + "Feature_Weights.png",dpi=600)
# plt.close()
plt.show()

## Running the next two lines shows us that genes that we know to be important for early human development, are amongst the highest
# ranked genes when considering the normalised network feature weights.
np.where(np.isin(np.argsort(-Feature_Weights),Known_Important_Gene_Inds))[0]
np.where(np.isin(np.argsort(-Normalised_Network_Feature_Weights),Known_Important_Gene_Inds))[0]

## Take the top 4000 ranked genes. 4000 is relitively arbitrary and there is some flexibility, but above 5500 genes, performance
# downstream drops significantly. For now, determining how many of the top ranked genes to choose is an intterative process for every
# dataset.
Use_Inds = np.argsort(-Normalised_Network_Feature_Weights)[np.arange(4000)] 
Selected_Genes = Subset_Used_Features[Use_Inds]
Selected_Genes.shape[0]
## Sometimes when picking how many of the top ranked genes to take, it is useful to see if important known
# markers are captured by your threshold.
np.isin(Known_Important_Genes,Selected_Genes)

## Now we will use our Masked_ESSs subsetted to the top 4000 cESFW weighted genes to cluster the genes into groups.
# We will use hdbscan to cluster the genes because it doesn't require the user to pre-define the number of clusters.
# However, as you will see in the next plot, using hdbscan is probably overkill, because we tend to find 2 distinct groups of genes.

clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
clusterer.fit(Masked_ESSs[np.ix_(Use_Inds,Use_Inds)])

## Extract hdbscan gene cluster labels
Gene_Labels = clusterer.labels_
Unique_Gene_Labels = np.unique(Gene_Labels)

## Visualise the gene clusters on a UMAP.
Neighbours = 20
Dist = 0.1
Gene_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Masked_ESSs[np.ix_(Use_Inds,Use_Inds)])

## What we tend to see in this plot, with this data and other scRNA-seq data, is that 2 main clusters appear. One of the clusters
# has on average, higher Normalised_Network_Feature_Weights, and is relitively "blobby"/lacks branching structure. The other main cluster
# has on average lower Normalised_Network_Feature_Weights, and contains branching struture, indicating genes that are locally connected,
# but globally disconnected. It is this second cluster of genes with branches that often produces high reoslution embeddings when these
# genes are used to interrogate the orignal scRNA-seq dataset. More interrogation of what makes these two clusters of genes distinct, but
# as an example of their difference we can look at the Gene Ontology (GO) enrichment for each cluster. The non-branching cluster tends to be
# enriched for genes such as ribosome regulation and catalysis, and RNA regulation. Conversly, for this human pre-implantation embryo
# dataset, the branching set of genes is enriched for GO terms related to developmental and differentiation processes.

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.title("Colour = hdbscan labels", fontsize=20)
plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=7,c=Gene_Labels)
plt.colorbar()
plt.xlabel("UMAP 1",fontsize=16)
plt.ylabel("UMAP 2",fontsize=16)
plt.subplot(1,2,2)
plt.title("Colour = Normalised_Network_Feature_Weights", fontsize=20)
plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=7,c=Feature_Weights[Use_Inds])
plt.colorbar()
plt.xlabel("UMAP 1",fontsize=16)
plt.ylabel("UMAP 2",fontsize=16)
# plt.savefig(human_embryo_path + "Gene_Cluster.png",dpi=600)
# plt.close()
plt.show()

## We will now subset the original scRNA-seq data down to the genes the branching cluster to create our high resolution embedding.

Cluster_Use_Gene_IDs = Selected_Genes[np.where(np.isin(Gene_Labels,np.array([0])))[0]]

Reduced_Input_Data = Human_Embryo_Counts[Cluster_Use_Gene_IDs]

Neighbours = 50
Dist = 0.1

Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
Embedding = Embedding_Model.embedding_

Plot_UMAP(Embedding,Human_Sample_Info)
plt.show()

## The above plot is an excellent start, with clear branching points and distinct cell types.
# However, there appears to still be a little bit of batch effects preventing the Yanagida dataset from integrating with the other
# datasets. We hypothesise that there might be a few genes remaining that are still overly enriched in the Yanagida and Xiang datasets.
# We now add one more gene filter, and remove any genes that are in our Cluster_Use_Gene_IDs set, that are also in the top 150 ESS ranked
# genes for the Xiang and Yanagida datasets.
Exclude_Xiang_2 = Used_Features[np.flip(np.argsort(np.asarray(Dataset_ESSs["Xiang 2020"])))[np.arange(150)]] 
Exclude_Yanagida_2 = Used_Features[np.flip(np.argsort(np.asarray(Dataset_ESSs["Yanagida 2020"])))[np.arange(150)]]

Exclude_2 = np.union1d(Exclude_Xiang_2,Exclude_Yanagida_2)

Use_Gene_IDs_2 = Cluster_Use_Gene_IDs[np.isin(Cluster_Use_Gene_IDs,Exclude_2)==0]
Cluster_Use_Gene_IDs.shape[0] - Use_Gene_IDs_2.shape[0]
## This filter only removes an additional 93 genes, but make a significant difference to the final plot, without having to smooth
# or augment the data in any way.

Reduced_Input_Data = Human_Embryo_Counts[Use_Gene_IDs_2]

Neighbours = 50
Dist = 0.1

Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
Embedding = Embedding_Model.embedding_

Plot_UMAP(Embedding,Human_Sample_Info)
# plt.savefig(human_embryo_path + "Meta_Data.png",dpi=600)
# plt.close()
plt.show()

## We shall also remove genes that have exceptionally high expression levels.

Means = np.sum(Human_Embryo_Counts[Use_Gene_IDs_2],axis=0) / np.sum(Human_Embryo_Counts[Use_Gene_IDs_2] > 0,axis=0)
Drop = Use_Gene_IDs_2[np.where(Means > 2000)[0]]
Use_Gene_IDs_3 = Use_Gene_IDs_2[np.isin(Use_Gene_IDs_2,Drop)==0]

Reduced_Input_Data = Human_Embryo_Counts[Use_Gene_IDs_3]

Neighbours = 50
Dist = 0.1

Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
Embedding = Embedding_Model.embedding_

Plot_UMAP(Embedding,Human_Sample_Info)
# plt.savefig(human_embryo_path + "Meta_Data.png",dpi=600)
# plt.close()
plt.show()

### Un-comment to save the final set of genes and the UMAP model.
# np.save(human_embryo_path+"Saved_cESFW_Genes.npy",Use_Gene_IDs_3)
# f_name = human_embryo_path+'Human_Embryo_Embedding_Model.sav'
# pickle.dump(Embedding_Model, open(f_name, 'wb'))

