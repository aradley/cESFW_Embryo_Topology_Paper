

### Dependencies ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import pickle
from sklearn.metrics import pairwise_distances
import plotly.express as px 
Colours = px.colors.qualitative.Dark24
Colours.remove('#222A2A')
Colours = np.concatenate((Colours,Colours))
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import cESFW
from scipy.cluster.hierarchy import ward, leaves_list
from scipy.spatial.distance import pdist
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_venn import venn3
from scipy.stats import gaussian_kde
from scipy.stats import zscore

### Dependencies ###

### 
def _forward(x):
    return np.sqrt(x)

def _inverse(x):
    return x**2

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

## Set path for retreiving and depositing data
human_embryo_path = "/mnt/c/Users/arthu/OneDrive - University of Exeter/Documents/cESFW_Paper/cESFW_Human_Embryo_Topology/Data/Human_Embryo_Analysis/"

Human_Sample_Info = pd.read_csv(human_embryo_path+"Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv(human_embryo_path+"Human_Embryo_Counts.csv",header=0,index_col=0)
Saved_cESFW_Genes = np.load(human_embryo_path + "Saved_cESFW_Genes.npy",allow_pickle=True)

f_name = human_embryo_path+'Human_Embryo_Embedding_Model.sav'
Embedding_Model = pickle.load((open(f_name, 'rb')))
Human_Embryo_Embedding = Embedding_Model.embedding_

# Human_Embryo_Embedding = np.load(human_embryo_path+"UMAP_Human_Embryo_Embedding.npy")

Plot_UMAP(Human_Embryo_Embedding,Human_Sample_Info)
plt.show()

### Cluster Samples ###

## Agglomerative Clustering
distmat = pairwise_distances(Human_Embryo_Counts[Saved_cESFW_Genes],metric='correlation')

clustering = AgglomerativeClustering(n_clusters=30).fit(distmat)
Cluster_Labels = clustering.labels_
Unique_Cluster_Labels = np.unique(Cluster_Labels)

## Annotate clusters based on marker genes
Manual_Annotations = np.zeros(Human_Sample_Info.shape[0]).astype("str")

Eight_cell = np.where(np.isin(Cluster_Labels,np.array([3])))[0]
Morula = np.where(np.isin(Cluster_Labels,np.array([25,15])))[0]
ICM_TE_Branch = np.where(np.isin(Cluster_Labels,np.array([18,17])))[0]
ICM = np.where(np.isin(Cluster_Labels,np.array([23,7])))[0]
Post_Epiblast = np.where(np.isin(Cluster_Labels,np.array([11])))[0]
Pre_Epiblast = np.where(np.isin(Cluster_Labels,np.array([19,16])))[0]
Epi_Hypo_Branch = np.where(np.isin(Cluster_Labels,np.array([10])))[0]
Hypoblast = np.where(np.isin(Cluster_Labels,np.array([24,12,6])))[0]
Early_TE = np.where(np.isin(Cluster_Labels,np.array([2])))[0]
Mid_TE = np.where(np.isin(Cluster_Labels,np.array([22,20,14,5,0])))[0]
Polar_TE = np.where(np.isin(Cluster_Labels,np.array([29,1])))[0]
Mural_TE = np.where(np.isin(Cluster_Labels,np.array([21,8])))[0]
sTB =  np.where(np.isin(Cluster_Labels,np.array([9])))[0]
cTB =  np.where(np.isin(Cluster_Labels,np.array([26,28])))[0]
ExE_Mech = np.where(np.isin(Cluster_Labels,np.array([27])))[0]
Unknown = np.where(np.isin(Cluster_Labels,np.array([13,4])))[0]
#
Manual_Annotations[Eight_cell] = "8-Cell"
Manual_Annotations[Morula] = "Morula"
Manual_Annotations[ICM_TE_Branch] = "ICM/TE Branch"
Manual_Annotations[ICM] = "ICM"
Manual_Annotations[Post_Epiblast] = "Embryonic Disc"
Manual_Annotations[Pre_Epiblast] = "preIm-Epi"
Manual_Annotations[Epi_Hypo_Branch] = "Epi/Hypo Branch"
Manual_Annotations[Hypoblast] = "Hypo"
Manual_Annotations[Unknown] = "Unknown"
Manual_Annotations[Early_TE] = "Early TE"
Manual_Annotations[Mid_TE] = "Mid TE"
Manual_Annotations[Polar_TE] = "Polar TE"
Manual_Annotations[Mural_TE] = "Mural TE"
Manual_Annotations[sTB] = "sTB"
Manual_Annotations[cTB] = "cTB"
Manual_Annotations[ExE_Mech] = "ExE-Mech"

Human_Sample_Info["Manual_Annotations"] = Manual_Annotations

#Human_Sample_Info.to_csv(human_embryo_path + "Human_Sample_Info.csv")

### Visualise the unsupervised clusters (in 3 seperate plots because there are 30 clusters).
## Plot 1
plt.figure(figsize=(2.33,2.33))
plt.title("Unsupervised cell clustering",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey",zorder=-1)
for i in np.arange(10):
    IDs = np.where(Cluster_Labels == Unique_Cluster_Labels[i])[0]
    plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c=Colours[i],label=i)

# lgnd = plt.legend(scatterpoints=1, fontsize=10, ncol=4,framealpha=1)
# for i in np.arange(Unique_Cluster_Labels.shape[0]):
#     lgnd.legend_handles[i]._sizes = [30]
    
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Embryo_Celltype_Unsupervised_Clusters_1.png",dpi=600)
plt.close()

## Plot 2
plt.figure(figsize=(2.33,2.33))
plt.title("Unsupervised cell clustering",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey",zorder=-1)
for i in np.arange(10,20):
    IDs = np.where(Cluster_Labels == Unique_Cluster_Labels[i])[0]
    plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c=Colours[i],label=i)

# lgnd = plt.legend(scatterpoints=1, fontsize=10, ncol=4,framealpha=1)
# for i in np.arange(Unique_Cluster_Labels.shape[0]):
#     lgnd.legend_handles[i]._sizes = [30]
    
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Embryo_Celltype_Unsupervised_Clusters_2.png",dpi=600)
plt.close()

## Plot 3
plt.figure(figsize=(2.33,2.33))
plt.title("Unsupervised cell clustering",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey",zorder=-1)
for i in np.arange(20,30):
    IDs = np.where(Cluster_Labels == Unique_Cluster_Labels[i])[0]
    plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c=Colours[i],label=i)

# lgnd = plt.legend(scatterpoints=1, fontsize=10, ncol=4,framealpha=1)
# for i in np.arange(Unique_Cluster_Labels.shape[0]):
#     lgnd.legend_handles[i]._sizes = [30]
    
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Embryo_Celltype_Unsupervised_Clusters_3.png",dpi=600)
plt.close()


### Visualise annotated clusters
plt.figure(figsize=(2.33,2.33))
plt.title("Cluster annotations",fontsize=10)
Manual_Annotations = np.asarray(Human_Sample_Info["Manual_Annotations"])
Unique_Manual_Annotations = np.unique(Manual_Annotations)
# Order the annotated cell types for plotting
Unique_Manual_Annotations = Unique_Manual_Annotations[np.array([0,9,7,6,3,14,2,4,5,1,8,10,13,11,15,12])]
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    if Unique_Manual_Annotations[i] == "Unknown":
        plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.25,label=Unique_Manual_Annotations[i],c="grey",zorder=-1)
    else:
        plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,label=Unique_Manual_Annotations[i],c=Colours[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Embryo_Celltype_Manual_Annotations.png",dpi=600)
plt.close()

plt.figure(figsize=(2.33,2.33))
plt.title("Stirparo et al. 2018",fontsize=10)
Stirparo_Labels = np.asarray(Human_Sample_Info["Stirparo_Labels"]).astype("str")
Unique_Stirparo_Labels = np.unique(Stirparo_Labels)
for i in np.arange(Unique_Stirparo_Labels.shape[0]):
    IDs = np.where(Stirparo_Labels == Unique_Stirparo_Labels[i])[0]
    if Unique_Stirparo_Labels[i] != "nan" and Unique_Stirparo_Labels[i] != 'Late Morula':
        z=1
        if Unique_Stirparo_Labels[i] == "TE":
            z = -1
        plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,label=Unique_Stirparo_Labels[i],zorder=z)
plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.25,c="grey",zorder=-1,label="nan")
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Stirparo_Labels.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.savefig(human_embryo_path + "Plots/" + "Embryo_Celltype_Stirparo.png",dpi=600)
plt.close()


plt.figure(figsize=(2.33,2.33))
plt.title("Timepoints",fontsize=10)
Timepoints = np.asarray(Human_Sample_Info["EmbryoDay"]).astype("f")
Unique_Timepoints = np.unique(Timepoints)
Unique_Timepoints = np.delete(Unique_Timepoints,np.where(np.isnan(Unique_Timepoints)))
for i in np.arange(Unique_Timepoints.shape[0]):
    IDs = np.where(Timepoints == Unique_Timepoints[i])[0]
    plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,label=Unique_Timepoints[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Timepoints.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Plots/" + "Embryo_Timepoints.png",dpi=800)
plt.close()
#

plt.figure(figsize=(2.33,2.33))
plt.title("Datasets",fontsize=10)
Datasets = np.asarray(Human_Sample_Info["Dataset"])
Unique_Datasets = np.unique(Datasets)
for i in np.arange(Unique_Datasets.shape[0]):
    IDs = np.where(Datasets == Unique_Datasets[i])[0]
    plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,label=Unique_Datasets[i])
#

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Datasets.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Plots/" + "Embryo_Datasets.png",dpi=800)
plt.close()
#

### HVG method embeddings

Embryo_Subset_Used_Features = np.asarray(pd.read_csv(human_embryo_path+"Embryo_Subset_Used_Features.txt",header=None,index_col=None).T)[0]

Scran_HVGs = np.asarray(pd.read_csv(human_embryo_path+"Embryo_SCRAN_HVG_Order.csv",header=0,index_col=0).T)[0]
Scran_HVGs = np.asarray(Embryo_Subset_Used_Features[Scran_HVGs-1])
Seurat_HVGs = np.asarray(pd.read_csv(human_embryo_path+"Embryo_Seurat_HVG_Order.csv",header=0,index_col=0).T)[0]
Seurat_HVGs = np.asarray(Embryo_Subset_Used_Features[Seurat_HVGs-1])
cESFW_Ranked_Genes = np.load(human_embryo_path+"Embryo_cESFW_ranked_genes.npy",allow_pickle=True)

Scran_Top_Genes = Scran_HVGs[np.arange(3012)]
Seurat_Top_Genes = Seurat_HVGs[np.arange(3012)]
Saved_cESFW_Genes

Int1 = np.intersect1d(Saved_cESFW_Genes,Seurat_Top_Genes).shape[0]
Int2 = np.intersect1d(Saved_cESFW_Genes,Scran_Top_Genes).shape[0]
Int3 = np.intersect1d(Seurat_Top_Genes,Scran_Top_Genes).shape[0]
Int4 = np.intersect1d(Saved_cESFW_Genes,Seurat_Top_Genes)
Int4 = np.intersect1d(Int4,Scran_Top_Genes).shape[0]

# Use the venn3 function
plt.figure(figsize=(5,5))
venn3(subsets = (Saved_cESFW_Genes.shape[0], Seurat_Top_Genes.shape[0], Int1, Scran_Top_Genes.shape[0], Int2, Int3, Int4), set_labels = ('cESFW workflow\ngenes', 'Seurat HVGs', 'Scran HVGs'))
plt.savefig(human_embryo_path + "Plots/" + "HVG_Venndiagram.png",dpi=800)
plt.close()
plt.show()


# Sil scores

Scran_Top_Genes
Seurat_Top_Genes
Saved_cESFW_Genes

from sklearn.metrics import silhouette_samples

cESFW_Sils = silhouette_samples(Human_Embryo_Counts[Saved_cESFW_Genes], Manual_Annotations, metric='correlation')
cESFW_Means = []
for i in range(Unique_Manual_Annotations.shape[0]):
    cESFW_Means.append(cESFW_Sils[Manual_Annotations == Unique_Manual_Annotations[i]].mean())
#
cESFW_Means = np.asarray(cESFW_Means)

Seurat_Sils = silhouette_samples(Human_Embryo_Counts[Seurat_Top_Genes], Manual_Annotations, metric='correlation')
Seurat_Means = []
for i in range(Unique_Manual_Annotations.shape[0]):
    Seurat_Means.append(Seurat_Sils[Manual_Annotations == Unique_Manual_Annotations[i]].mean())
#
Seurat_Means = np.asarray(Seurat_Means)

Scran_Sils = silhouette_samples(Human_Embryo_Counts[Scran_Top_Genes], Manual_Annotations, metric='correlation')
Scran_Means = []
for i in range(Unique_Manual_Annotations.shape[0]):
    Scran_Means.append(Scran_Sils[Manual_Annotations == Unique_Manual_Annotations[i]].mean())
#
Scran_Means = np.asarray(Scran_Means)


All_Sil_Scores = np.concatenate((cESFW_Means,Scran_Means,Seurat_Means))
All_Sil_Labels = np.concatenate((Unique_Manual_Annotations,Unique_Manual_Annotations,Unique_Manual_Annotations))
All_Sil_Methods = np.concatenate((np.repeat("cESFW workflow genes",Unique_Manual_Annotations.shape[0]),np.repeat("Scran HVGs",Unique_Manual_Annotations.shape[0]),np.repeat("Seurat HVGs",Unique_Manual_Annotations.shape[0])))

Sil_DF = pd.DataFrame([All_Sil_Methods,All_Sil_Labels,All_Sil_Scores],index=np.array(["Method","Cell type","Silhouette scores"])).T

plt.figure(figsize=(5,3.5))
#pal = sns.color_palette("Greys_d", PR_AUCs.shape[0])
ax = sns.barplot(data=Sil_DF, y="Cell type", x="Silhouette scores", hue="Method",palette=np.array(["#ff9999","#99cc99","#9999ff"]))
ax.set(xlabel="Silhouette scores",fontsize=10)
ax.set(ylabel=None)
plt.legend([],[], frameon=False)
#plt.yticks([])
plt.xlim(-0.31,0.55)
plt.title("Cell type Silhouette scores",fontsize=12)
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    plt.gca().get_yticklabels()[i].set_color(Colours[i])
plt.tight_layout()
plt.savefig(human_embryo_path + "Plots/" + "Cell_Type_Silhoutte_Scores.png",dpi=600)
plt.close()


plt.show()


####

Reduced_Input_Data = Human_Embryo_Counts[Scran_Top_Genes]

Neighbours = 50
Dist = 0.1

Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
Embedding = Embedding_Model.embedding_

Plot_UMAP(Embedding,Human_Sample_Info)
plt.show()

plt.figure(figsize=(2.33,2.33))
plt.title("Unsupervised cluster annotations",fontsize=10)
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    if Unique_Manual_Annotations[i] == "Unknown":
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.25,label=Unique_Manual_Annotations[i],c="grey",zorder=-1)
    else:
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.5,label=Unique_Manual_Annotations[i],c=Colours[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Scran_Celltype_Manual_Annotations.png",dpi=600)
plt.close()


plt.figure(figsize=(2.33,2.33))
plt.title("Timepoints",fontsize=10)
Timepoints = np.asarray(Human_Sample_Info["EmbryoDay"]).astype("f")
Unique_Timepoints = np.unique(Timepoints)
Unique_Timepoints = np.delete(Unique_Timepoints,np.where(np.isnan(Unique_Timepoints)))
for i in np.arange(Unique_Timepoints.shape[0]):
    IDs = np.where(Timepoints == Unique_Timepoints[i])[0]
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.5,label=Unique_Timepoints[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Timepoints.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Plots/" + "Scran_Timepoints.png",dpi=800)
plt.close()

plt.figure(figsize=(2.33,2.33))
plt.title("Datasets",fontsize=10)
Datasets = np.asarray(Human_Sample_Info["Dataset"])
Unique_Datasets = np.unique(Datasets)
for i in np.arange(Unique_Datasets.shape[0]):
    IDs = np.where(Datasets == Unique_Datasets[i])[0]
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.5,label=Unique_Datasets[i])
#

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Datasets.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Plots/" + "Scran_Datasets.png",dpi=800)
plt.close()


###


Reduced_Input_Data = Human_Embryo_Counts[Seurat_Top_Genes]

Neighbours = 50
Dist = 0.1

Embedding_Model = umap.UMAP(n_neighbors=Neighbours, metric='correlation', min_dist=Dist, n_components=2).fit(Reduced_Input_Data)
Embedding = Embedding_Model.embedding_

Plot_UMAP(Embedding,Human_Sample_Info)
plt.show()

plt.figure(figsize=(2.33,2.33))
plt.title("Unsupervised cluster annotations",fontsize=10)
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    if Unique_Manual_Annotations[i] == "Unknown":
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.25,label=Unique_Manual_Annotations[i],c="grey",zorder=-1)
    else:
        plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.5,label=Unique_Manual_Annotations[i],c=Colours[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Seurat_Celltype_Manual_Annotations.png",dpi=600)
plt.close()


plt.figure(figsize=(2.33,2.33))
plt.title("Timepoints",fontsize=10)
Timepoints = np.asarray(Human_Sample_Info["EmbryoDay"]).astype("f")
Unique_Timepoints = np.unique(Timepoints)
Unique_Timepoints = np.delete(Unique_Timepoints,np.where(np.isnan(Unique_Timepoints)))
for i in np.arange(Unique_Timepoints.shape[0]):
    IDs = np.where(Timepoints == Unique_Timepoints[i])[0]
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.5,label=Unique_Timepoints[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Timepoints.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Plots/" + "Seurat_Timepoints.png",dpi=800)
plt.close()

plt.figure(figsize=(2.33,2.33))
plt.title("Datasets",fontsize=10)
Datasets = np.asarray(Human_Sample_Info["Dataset"])
Unique_Datasets = np.unique(Datasets)
for i in np.arange(Unique_Datasets.shape[0]):
    IDs = np.where(Datasets == Unique_Datasets[i])[0]
    plt.scatter(Embedding[IDs,0],Embedding[IDs,1],s=0.5,label=Unique_Datasets[i])
#

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Datasets.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Plots/" + "Seurat_Datasets.png",dpi=800)
plt.close()



####### Gene expression heatmap #######

Used_Features = np.load(human_embryo_path + "Used_Features.npy",allow_pickle=True)

Annotation_ESSs = np.zeros((Used_Features.shape[0],Unique_Manual_Annotations.shape[0]))
Annotation_EPs = np.zeros((Used_Features.shape[0],Unique_Manual_Annotations.shape[0]))

for i in np.arange(Unique_Manual_Annotations.shape[0]):
    Annotation_Labels = np.zeros(Human_Sample_Info.shape[0])
    Annotation_Labels[np.where(Manual_Annotations==Unique_Manual_Annotations[i])[0]] = 1
    # If the number of samples in the dataset is greater than half the cardinality of the entire data, flip the majority/minority state labels.
    if np.sum(Annotation_Labels) > (Human_Sample_Info.shape[0]/2):
        Annotation_Labels = (Annotation_Labels * - 1) + 1
    ## Calulate the ESSs and EPs of the dataset with every feature/gene in the scaled matrix.
    Individual_ESSs, Individual_EPs = cESFW.Calculate_Individual_ESS_EPs(Annotation_Labels,human_embryo_path)
    ## Create a ranked list of features the are enriched in the dataset
    Annotation_ESSs[:,i] = Individual_ESSs
    Annotation_EPs[:,i] = Individual_EPs

Annotation_ESSs = pd.DataFrame(Annotation_ESSs,columns=Unique_Manual_Annotations,index=Used_Features)
Annotation_EPs = pd.DataFrame(Annotation_EPs,columns=Unique_Manual_Annotations,index=Used_Features)

Ranked_Genes = pd.DataFrame(np.zeros(Annotation_ESSs.shape),columns=Annotation_ESSs.columns)

for i in np.arange(Ranked_Genes.shape[1]):
    Ranked_Genes[Annotation_ESSs.columns[i]] = np.asarray(Annotation_ESSs.index[np.argsort(-Annotation_ESSs[Annotation_ESSs.columns[i]])])

Ranked_Genes.to_csv(human_embryo_path+"Ranked_Genes.csv")
#Annotation_ESSs.to_csv(human_embryo_path+"Annotation_ESSs.csv")

###

Gene_Annotation_Memberships = Unique_Manual_Annotations[np.argmax(np.asarray(Annotation_ESSs.loc[Saved_cESFW_Genes]),axis=1)]

Plot_Colours = np.asarray(Colours)[np.arange(Unique_Manual_Annotations.shape[0])]
lut = dict(zip(Unique_Manual_Annotations,Plot_Colours))
row_colors = pd.DataFrame(Manual_Annotations,index=Human_Embryo_Counts.index)[0].map(lut)

Columns_Order = []
Append_Plot_Colours = []
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    Annotation_Genes = np.where(Gene_Annotation_Memberships == Unique_Manual_Annotations[i])[0]
    Z = ward(pdist(Human_Embryo_Counts[Saved_cESFW_Genes[Annotation_Genes]].T))
    Columns_Order.append(Annotation_Genes[leaves_list(Z)])
    Append_Plot_Colours.append(np.repeat(Plot_Colours[i],Annotation_Genes.shape[0]))

Columns_Order_Genes = Saved_cESFW_Genes[np.concatenate(Columns_Order)]
Append_Plot_Colours = np.concatenate(Append_Plot_Colours)

Spacing = 8
Appended_Row_Chunks = np.empty((0,Columns_Order_Genes.shape[0]))
Append_Row_Indexs = np.empty((0,Human_Embryo_Counts.shape[0]))
Append_Row_Colours = np.empty((0,Human_Embryo_Counts.shape[0]))
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    Annotation_Label = Unique_Manual_Annotations[i]
    Annotation_IDs = Human_Embryo_Counts.index[np.where(Manual_Annotations == Annotation_Label)[0]]
    Append_Row_Indexs = np.append(Append_Row_Indexs,Annotation_IDs)
    Append_Row_Colours = np.append(Append_Row_Colours,np.asarray(row_colors.loc[Annotation_IDs]))
    #
    Z = ward(pdist(Human_Embryo_Counts[Saved_cESFW_Genes].loc[Annotation_IDs],metric="correlation"))
    Appended_Row_Chunks = np.row_stack((Appended_Row_Chunks,Human_Embryo_Counts.loc[Annotation_IDs[leaves_list(Z)]][Columns_Order_Genes]))
    #
    Append_Row_Indexs = np.append(Append_Row_Indexs,["None"]*Spacing)
    Append_Row_Colours = np.append(Append_Row_Colours,["None"]*Spacing)
    Annotation_Break = np.empty((Spacing,Columns_Order_Genes.shape[0],))
    Annotation_Break[:] = np.nan
    Appended_Row_Chunks = np.row_stack((Appended_Row_Chunks,Annotation_Break))


for i in np.arange(Unique_Manual_Annotations.shape[0]):
    Inds = np.where(Append_Plot_Colours == Plot_Colours[i])[0]
    Lower = Inds[0]
    Upper = Inds[-1]
    #
    Annotation_Break = np.empty((Appended_Row_Chunks.shape[0],Spacing,))
    Annotation_Break[:] = np.nan
    Appended_Row_Chunks = np.hstack((Appended_Row_Chunks[:,:Upper], Annotation_Break, Appended_Row_Chunks[:,Upper:]))
    #
    Append_Plot_Colours = np.concatenate((Append_Plot_Colours[:Upper], ["None"] * Spacing,  Append_Plot_Colours[Upper:]))
    Columns_Order_Genes = np.concatenate((Columns_Order_Genes[:Upper], ["None"] * Spacing,  Columns_Order_Genes[Upper:]))


Appended_Row_Chunks = np.log2(Appended_Row_Chunks+1)
#Appended_Row_Chunks = (Appended_Row_Chunks - np.nanmean(Appended_Row_Chunks,axis=0)) / np.nanstd(Appended_Row_Chunks,axis=0)
Appended_Row_Chunks = Appended_Row_Chunks / np.nanmax(Appended_Row_Chunks,axis=0)

norm = matplotlib.colors.FuncNorm((_forward, _inverse), vmin=0, vmax=1)
# sns.clustermap(Appended_Row_Chunks,row_cluster=False,col_cluster=False,cmap="seismic",norm=norm,row_colors=Append_Row_Colours,col_colors=Append_Plot_Colours)
# plt.show()

Keep_Rows = np.where(Append_Row_Colours != Plot_Colours[-1])[0]
Keep_Columns = np.where(Append_Plot_Colours != Plot_Colours[-1])[0]
Appended_Row_Chunks = Appended_Row_Chunks[np.ix_(Keep_Rows,Keep_Columns)]
Append_Plot_Colours = Append_Plot_Colours[Keep_Columns]
Columns_Order_Genes = Columns_Order_Genes[Keep_Columns]

# Naive epiblast: LEFTY2, STAT3, HES1, SOCS3, MMP2, GBX2, LEF1, KLF2, ZIC3. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5413953/
# Primed epiblast: FGF2, SFRP2, DUSP8, CACNA1A, HGF, CER1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5413953/
# Mural TE: HAND1. https://www.science.org/doi/10.1126/sciadv.abj3725
# Polar TE: CCR7, CYP19A1, DLX5, MUC15. https://www.science.org/doi/10.1126/sciadv.abj3725
# 8-cell: ZSCAN4, TPRX1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8901440/
# Hypoblast: FOXA2. https://doi.org/10.1242/dev.201522
# Early Hypoblast: LGALS2. https://doi.org/10.1242/dev.201522
# ICM: SPIC. 
# Morula: DPRX. https://pubmed.ncbi.nlm.nih.gov/29361568/ SOX2. https://www.sciencedirect.com/science/article/pii/S1534580710001103


#Genes_of_Interest = np.array(["HAND1","FGF2","CYP19A1","LEFTY2","ZSCAN4","FOXA2","LGALS2","PRSS3","SOX2","DPRX"])
Genes_of_Interest = np.array(["ZSCAN4","DPRX","BAIAP2L2","SPIC","LEFTY1","SFRP2","LGALS2","FLRT3","CYP26A1","RGS13","HAND1","ISM2","GAST","SLC38A3","HGF"])
Genes_of_Interest[np.isin(Genes_of_Interest,Columns_Order_Genes)==0]

X_Ticks = Columns_Order_Genes.copy()
X_Ticks[np.where(np.isin(Columns_Order_Genes,Genes_of_Interest)==0)[0]] = None

cm = sns.clustermap(Appended_Row_Chunks,row_cluster=False,col_cluster=False,cmap="seismic",norm=norm,row_colors=Append_Row_Colours[Keep_Rows],col_colors=Append_Plot_Colours,yticklabels=False,xticklabels=False,figsize=(9,5))
#plt.setp(cm.ax_heatmap.get_xticklabels(), rotation=45)
plt.savefig(human_embryo_path + "Plots/" + "cESFW_Heatmap.png",dpi=800)
plt.close()
plt.show()

# Naive epiblast: LEFTY1, LEFTY2. https://stemcellsjournals.onlinelibrary.wiley.com/doi/10.1002/stem.2071
# Primed epiblast: FGF2, SFRP2, DUSP8, CACNA1A, HGF, CER1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5413953/
# Mural TE: HAND1. https://www.science.org/doi/10.1126/sciadv.abj3725
# Polar TE: CCR7, CYP19A1, DLX5, MUC15. https://www.science.org/doi/10.1126/sciadv.abj3725
# 8-cell: ZSCAN4, TPRX1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8901440/
# Hypoblast: FOXA2. https://doi.org/10.1242/dev.201522
# Early Hypoblast: LGALS2. https://doi.org/10.1242/dev.201522
# ICM: PRSS3, SPIC. 
# Morula: DPRX. https://pubmed.ncbi.nlm.nih.gov/29361568/ SOX2. https://www.sciencedirect.com/science/article/pii/S1534580710001103

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("SPIC (ICM)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["SPIC"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "SPIC.png",dpi=800)
plt.close()

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("FOXA2 (Hypoblast)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["FOXA2"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "FOXA2.png",dpi=800)
plt.close()


fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("NODAL (Epiblast)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["NODAL"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "NODAL.png",dpi=800)
plt.close()


######## Overlay samples onto embedding plots ########

naive_primed_path = "/mnt/c/Users/arthu/OneDrive - University of Exeter/Documents/cESFW_Paper/Naive_Primed_Data/"

# Load Messmer2019 data
Messmer_Sample_Info = pd.read_csv(naive_primed_path+"E-MTAB-6819_Sample_Info.csv",header=0,index_col=0)
Messmer_Sample_Info = Messmer_Sample_Info[["Assay Name","Factor Value[phenotype]"]]
Messmer_Sample_Info = Messmer_Sample_Info.drop_duplicates()
Messmer_Sample_Info.index = Messmer_Sample_Info["Assay Name"]

Messmer2019_Counts = pd.read_csv(naive_primed_path+"Messmet2019_sc_Counts.csv",header=0,index_col=0)

Present = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Messmer2019_Counts.columns)]
Messmer2019_Counts = Messmer2019_Counts[Present]
Absent = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Messmer2019_Counts.columns)==0]
Absent = pd.DataFrame(np.zeros((Messmer2019_Counts.shape[0],Absent.shape[0])),index=Messmer2019_Counts.index,columns=Absent)
Messmer2019_Counts = pd.concat([Messmer2019_Counts,Absent],axis=1)
Messmer2019_Counts = Messmer2019_Counts.loc[:,~Messmer2019_Counts.columns.duplicated()].copy()

# Load Guo2021 single cell data
Guo2021_sc_Counts = pd.read_csv(naive_primed_path+"Guo2021_sc_Counts.csv",header=0,index_col=0)
Ge2021_sc_Sample_Labels = np.asarray(Guo2021_sc_Counts.index).copy()
sep = "_"
for i in np.arange(Ge2021_sc_Sample_Labels.shape[0]):
    Ge2021_sc_Sample_Labels[i] = Ge2021_sc_Sample_Labels[i].split(sep, 1)[0]

Present = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Guo2021_sc_Counts.columns)]
Guo2021_sc_Counts = Guo2021_sc_Counts[Present]
Absent = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Guo2021_sc_Counts.columns)==0]
Absent = pd.DataFrame(np.zeros((Guo2021_sc_Counts.shape[0],Absent.shape[0])),index=Guo2021_sc_Counts.index,columns=Absent)
Guo2021_sc_Counts = pd.concat([Guo2021_sc_Counts,Absent],axis=1)
Guo2021_sc_Counts = Guo2021_sc_Counts.loc[:,~Guo2021_sc_Counts.columns.duplicated()].copy()

# Load Gao2019 cell data
Gao2019_Counts = pd.read_csv(naive_primed_path+"HEPSC_Counts.csv",header=0,index_col=0)
Gao2019_Sample_Info = pd.read_csv(naive_primed_path+"HEPSC_Sample_Info.csv",header=0,index_col=0)

Present = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Gao2019_Counts.columns)]
Gao2019_Counts = Gao2019_Counts[Present]
Absent = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Gao2019_Counts.columns)==0]
Absent = pd.DataFrame(np.zeros((Gao2019_Counts.shape[0],Absent.shape[0])),index=Gao2019_Counts.index,columns=Absent)
Gao2019_Counts = pd.concat([Gao2019_Counts,Absent],axis=1)
Gao2019_Counts = Gao2019_Counts.loc[:,~Gao2019_Counts.columns.duplicated()].copy()

# Load Mazid2022 cell data
Mazid2022_Counts = pd.read_csv(naive_primed_path+"Mazid2022_sc_Counts.csv",header=0,index_col=0)
Mazid2022_Counts = Mazid2022_Counts.T

Mazid2022_Sample_Labels = np.asarray(Mazid2022_Counts.index)
sep = "_"
for i in np.arange(Mazid2022_Sample_Labels.shape[0]):
    Mazid2022_Sample_Labels[i] = Mazid2022_Sample_Labels[i].split(sep, 1)[0]

Present = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Mazid2022_Counts.columns)]
Mazid2022_Counts = Mazid2022_Counts[Present]
Absent = Saved_cESFW_Genes[np.isin(Saved_cESFW_Genes,Mazid2022_Counts.columns)==0]
Absent = pd.DataFrame(np.zeros((Mazid2022_Counts.shape[0],Absent.shape[0])),index=Mazid2022_Counts.index,columns=Absent)
Mazid2022_Counts = pd.concat([Mazid2022_Counts,Absent],axis=1)
Mazid2022_Counts = Mazid2022_Counts.loc[:,~Mazid2022_Counts.columns.duplicated()].copy()


###

Plot_Celltypes = np.array(['8-Cell','preIm-Epi', 'Embryonic Disc'])
plt.figure(figsize=(2.33,2.33))
plt.title("Cluster annotations",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.1,c="grey")
for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    if np.sum(np.isin(Unique_Manual_Annotations[i],Plot_Celltypes)) == 1:
        plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,label=Unique_Manual_Annotations[i],c=Colours[i])

# lgnd2 = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd2.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "Primed_Naive_Annotations.png",dpi=600)
plt.close()


fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("FBP1 (preIm-Epi epiblast)",fontsize=10)
IDs = np.arange(Human_Embryo_Embedding.shape[0])
np.random.shuffle(IDs)
cmap_plot = plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c=np.log2(Human_Embryo_Counts["FBP1"]+1)[IDs],cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.58,0.05,0.4)))
cb.set_label('$log_2$(Expression)', labelpad=-35,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "FBP1.png",dpi=800)
plt.close()


fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("SFRP2 (Embryonic disc)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["SFRP2"]+1),cmap="seismic")
plt.show()
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.58,0.05,0.4)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "SFRP2.png",dpi=800)
plt.close()

####

f_name = human_embryo_path+'Human_Embryo_Embedding_Model.sav'
Embedding_Model = pickle.load((open(f_name, 'rb')))
Human_Embryo_Embedding = Embedding_Model.embedding_

Messmer2019_Overlay = Embedding_Model.transform(Messmer2019_Counts[Saved_cESFW_Genes])
Guo2021_sc_Overlay = Embedding_Model.transform(Guo2021_sc_Counts.iloc[np.where(Ge2021_sc_Sample_Labels == "PXGL")[0]][Saved_cESFW_Genes])
Gao2019_sc_Overlay = Embedding_Model.transform(Gao2019_Counts[Saved_cESFW_Genes])
Mazid2022_sc_Overlay = Embedding_Model.transform(Mazid2022_Counts[Saved_cESFW_Genes])

####

plt.figure(figsize=(2.33,2.33))
plt.title("Cultured single cell RNA samples",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey")
#
Inds = np.where(Messmer_Sample_Info["Factor Value[phenotype]"] == "naive")[0]
xy = np.vstack([Messmer2019_Overlay[Inds,0],Messmer2019_Overlay[Inds,1]])
Density = gaussian_kde(xy)(xy)
Density = Density/np.max(Density)
plt.scatter(Messmer2019_Overlay[Inds,0],Messmer2019_Overlay[Inds,1],s=0.5,label="Messmer 2019",c="tab:olive",alpha=Density)
#
xy = np.vstack([Guo2021_sc_Overlay[:,0],Guo2021_sc_Overlay[:,1]])
Density = gaussian_kde(xy)(xy)
Density = Density/np.max(Density)
plt.scatter(Guo2021_sc_Overlay[:,0],Guo2021_sc_Overlay[:,1],s=0.5,label="Guo 2021",c="tab:cyan",alpha=Density)
#
Inds = np.where(Messmer_Sample_Info["Factor Value[phenotype]"] == "primed")[0]
xy = np.vstack([Messmer2019_Overlay[Inds,0],Messmer2019_Overlay[Inds,1]])
Density = gaussian_kde(xy)(xy)
Density = Density/np.max(Density)
plt.scatter(Messmer2019_Overlay[Inds,0],Messmer2019_Overlay[Inds,1],s=0.5,label="Messmer 2019",c="tab:pink",alpha=Density)

# lgnd = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "sc_Naive_Primed_1.png",dpi=800)
plt.close()



plt.figure(figsize=(2.33,2.33))
plt.title("Cultured single cell RNA samples",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey")
Unique_Mazid_Sample_Labels = np.unique(Mazid2022_Sample_Labels)
for i in np.arange(Unique_Mazid_Sample_Labels.shape[0]):
    IDs = np.where(Mazid2022_Sample_Labels == Unique_Mazid_Sample_Labels[i])[0]
    xy = np.vstack([Mazid2022_sc_Overlay[IDs,0],Mazid2022_sc_Overlay[IDs,1]])
    Density = gaussian_kde(xy)(xy)
    Density = Density/np.max(Density)
    if Unique_Mazid_Sample_Labels[i] != "8CLC":
        plt.scatter(Mazid2022_sc_Overlay[IDs,0],Mazid2022_sc_Overlay[IDs,1],label=Unique_Mazid_Sample_Labels[i],s=0.5,alpha=Density)

#plt.legend()
# lgnd = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "sc_Naive_Primed_2.png",dpi=800)
plt.close()

plt.show()


plt.figure(figsize=(2.33,2.33))
plt.title("Cultured single cell RNA samples",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey")
#
IDs = np.arange(Gao2019_sc_Overlay.shape[0])
xy = np.vstack([Gao2019_sc_Overlay[IDs,0],Gao2019_sc_Overlay[IDs,1]])
Density = gaussian_kde(xy)(xy)
Density = Density/np.max(Density)
plt.scatter(Gao2019_sc_Overlay[IDs,0],Gao2019_sc_Overlay[IDs,1],s=0.5,alpha=Density,c="tab:purple")

# lgnd = plt.legend(scatterpoints=1, fontsize=10, ncol=2,framealpha=1)
# for i in np.arange(Unique_Manual_Annotations.shape[0]):
#     lgnd.legend_handles[i]._sizes = [30]

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "Plots/" + "sc_Naive_Primed_3.png",dpi=800)
plt.close()




from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


Manual_Annotations = np.asarray(Human_Sample_Info["Manual_Annotations"])
Training_Annotations = Manual_Annotations.copy()
Training_Human_Embryo_Embedding = Human_Embryo_Embedding.copy()
Keep_Labels = np.isin(Training_Annotations,np.array(['Unknown'])) == 0
Training_Annotations = Training_Annotations[Keep_Labels]
Training_Human_Embryo_Embedding = Training_Human_Embryo_Embedding[Keep_Labels,:]

celltype_knn = KNeighborsClassifier(n_neighbors=20).fit(Training_Human_Embryo_Embedding, Training_Annotations)

Theunissen2016_Predictions = celltype_knn.predict(Theunissen2016_Overlay)
Guo2021_sc_Predictions = celltype_knn.predict(Guo2021_sc_Overlay)
Gao2019_sc_Predictions = celltype_knn.predict(Gao2019_sc_Overlay)
Mazid2022_sc_Predictions = celltype_knn.predict(Mazid2022_sc_Overlay)
Messmer2019_sc_Predictions = celltype_knn.predict(Messmer2019_Overlay)


##### Curate data for plotting

Sub_Common_Genes = np.intersect1d(Human_Embryo_Counts.columns,Messmer2019_Counts.columns)
Sub_Common_Genes = np.intersect1d(Sub_Common_Genes,Guo2021_sc_Counts.columns)
Sub_Common_Genes = np.intersect1d(Sub_Common_Genes,Gao2019_Counts.columns)
Sub_Common_Genes = np.intersect1d(Sub_Common_Genes,Mazid2022_Counts.columns)

Sub_Human_Embryo_Counts = np.log2(Human_Embryo_Counts[Sub_Common_Genes]+1)#[Saved_cESFW_Genes]+1)
Sub_Messmer2019_Counts = np.log2(Messmer2019_Counts[Sub_Common_Genes]+1)#[Saved_cESFW_Genes]+1)
Sub_Guo2021_sc_Counts = np.log2(Guo2021_sc_Counts[Sub_Common_Genes].iloc[np.where(Ge2021_sc_Sample_Labels == "PXGL")[0]]+1)#[Saved_cESFW_Genes]+1)
Sub_Gao2019_Counts = np.log2(Gao2019_Counts[Sub_Common_Genes]+1)#[Saved_cESFW_Genes]+1)
Sub_Mazid2022_Counts = np.log2(Mazid2022_Counts[Sub_Common_Genes]+1)#[Saved_cESFW_Genes]+1)


Upper = np.percentile(Sub_Human_Embryo_Counts,97.5,axis=0)
Upper[Upper==0] = np.max(Sub_Human_Embryo_Counts,axis=0)[Upper==0]
Sub_Human_Embryo_Counts = Sub_Human_Embryo_Counts.clip(upper=Upper,axis=1)
#
Upper = np.percentile(Sub_Messmer2019_Counts,97.5,axis=0)
Upper[Upper==0] = np.max(Sub_Messmer2019_Counts,axis=0)[Upper==0]
Sub_Messmer2019_Counts = Sub_Messmer2019_Counts.clip(upper=Upper,axis=1)
#
Upper = np.percentile(Sub_Guo2021_sc_Counts,97.5,axis=0)
Upper[Upper==0] = np.max(Sub_Guo2021_sc_Counts,axis=0)[Upper==0]
Sub_Guo2021_sc_Counts = Sub_Guo2021_sc_Counts.clip(upper=Upper,axis=1)
#
Upper = np.percentile(Sub_Gao2019_Counts,97.5,axis=0)
Upper[Upper==0] = np.max(Sub_Gao2019_Counts,axis=0)[Upper==0]
Sub_Gao2019_Counts = Sub_Gao2019_Counts.clip(upper=Upper,axis=1)
#
Upper = np.percentile(Sub_Mazid2022_Counts,97.5,axis=0)
Upper[Upper==0] = np.max(Sub_Mazid2022_Counts,axis=0)[Upper==0]
Sub_Mazid2022_Counts = Sub_Mazid2022_Counts.clip(upper=Upper,axis=1)

Human_Factor = np.percentile(np.asarray(Sub_Human_Embryo_Counts)[Sub_Human_Embryo_Counts!=0],97.5)
Messmer_Factor = np.percentile(np.asarray(Sub_Messmer2019_Counts)[Sub_Messmer2019_Counts!=0],97.5)
Guo2021_sc_Factor = np.percentile(np.asarray(Sub_Guo2021_sc_Counts)[Sub_Guo2021_sc_Counts!=0],97.5)
Gao_Factor = np.percentile(np.asarray(Sub_Gao2019_Counts)[Sub_Gao2019_Counts!=0],97.5)
Mazid_Factor = np.percentile(np.asarray(Sub_Mazid2022_Counts)[Sub_Mazid2022_Counts!=0],97.5)

Sub_Human_Embryo_Counts = (Sub_Human_Embryo_Counts * (Guo2021_sc_Factor / Human_Factor)) / Guo2021_sc_Factor
Sub_Messmer2019_Counts = (Sub_Messmer2019_Counts * (Guo2021_sc_Factor / Messmer_Factor)) / Guo2021_sc_Factor
Sub_Guo2021_sc_Counts = (Sub_Guo2021_sc_Counts * (Guo2021_sc_Factor / Guo2021_sc_Factor)) / Guo2021_sc_Factor
Sub_Gao2019_Counts = (Sub_Gao2019_Counts * (Guo2021_sc_Factor / Mazid_Factor)) / Guo2021_sc_Factor
Sub_Mazid2022_Counts = (Sub_Mazid2022_Counts * (Guo2021_sc_Factor / Mazid_Factor)) / Guo2021_sc_Factor

plt.hist(np.asarray(Sub_Human_Embryo_Counts)[Sub_Human_Embryo_Counts!=0],bins=30,alpha=0.5,density=True)
plt.hist(np.asarray(Sub_Messmer2019_Counts)[Sub_Messmer2019_Counts!=0],bins=30,alpha=0.5,density=True)
plt.hist(np.asarray(Sub_Guo2021_sc_Counts)[Sub_Guo2021_sc_Counts!=0],bins=30,alpha=0.5,density=True)
plt.hist(np.asarray(Sub_Gao2019_Counts)[Sub_Gao2019_Counts!=0],bins=30,alpha=0.5,density=True)
plt.hist(np.asarray(Sub_Mazid2022_Counts)[Sub_Mazid2022_Counts!=0],bins=30,alpha=0.5,density=True)
plt.show()

###




#Manual_Annotations = np.asarray(Human_Sample_Info["Manual_Annotations"])
#Unique_Manual_Annotations = np.unique(Manual_Annotations)
#Unique_Manual_Annotations = Unique_Manual_Annotations[np.array([0,8,6,5,14,3,11,2,4,1,7,9,13,10,15,12])]


Orig_Sample_IDs = Messmer_Sample_Info["Factor Value[phenotype]"].copy()
Unique_Orig_Sample_IDs = np.unique(Orig_Sample_IDs)
Messmer2019_Ratios = pd.DataFrame(np.zeros((Unique_Orig_Sample_IDs.shape[0],Unique_Manual_Annotations.shape[0])),columns=Unique_Manual_Annotations)
for i in np.arange(Unique_Orig_Sample_IDs.shape[0]):
    Orig_Inds = np.where(Orig_Sample_IDs == Unique_Orig_Sample_IDs[i])[0]
    Predictions = Messmer2019_sc_Predictions[Orig_Inds]
    Unique_Predictions = np.unique(Predictions,return_counts=True)
    Unique_Prediction_Counts = Unique_Predictions[1]
    Unique_Predictions = Unique_Predictions[0]
    for j in np.arange(Unique_Predictions.shape[0]):
        Inds = np.where(Predictions == Unique_Predictions[j])[0]
        Ratio = Inds.shape[0] / Predictions.shape[0]
        Messmer2019_Ratios[Unique_Predictions[j]][i] = Ratio

Messmer2019_Ratios.index = np.array(["Messmer 2019\nNaive (t2iLG)","Messmer 2019\nPrimed (E8)"])

#
Orig_Sample_IDs = np.repeat("Naive",Guo2021_sc_Predictions.shape[0])
Unique_Orig_Sample_IDs = np.unique(Orig_Sample_IDs)
Guo2021_sc_Ratios = pd.DataFrame(np.zeros((Unique_Orig_Sample_IDs.shape[0],Unique_Manual_Annotations.shape[0])),columns=Unique_Manual_Annotations)
for i in np.arange(Unique_Orig_Sample_IDs.shape[0]):
    Orig_Inds = np.where(Orig_Sample_IDs == Unique_Orig_Sample_IDs[i])[0]
    Predictions = Guo2021_sc_Predictions[Orig_Inds]
    Unique_Predictions = np.unique(Predictions,return_counts=True)
    Unique_Prediction_Counts = Unique_Predictions[1]
    Unique_Predictions = Unique_Predictions[0]
    for j in np.arange(Unique_Predictions.shape[0]):
        Inds = np.where(Predictions == Unique_Predictions[j])[0]
        Ratio = Inds.shape[0] / Predictions.shape[0]
        Guo2021_sc_Ratios[Unique_Predictions[j]][i] = Ratio

Guo2021_sc_Ratios.index = np.array(["Guo 2021\nNaive (PGXL)"])

Orig_Sample_IDs = np.repeat("8-Cell",Gao2019_sc_Predictions.shape[0])
Unique_Orig_Sample_IDs = np.unique(Orig_Sample_IDs)
Gao2019_Ratios = pd.DataFrame(np.zeros((Unique_Orig_Sample_IDs.shape[0],Unique_Manual_Annotations.shape[0])),columns=Unique_Manual_Annotations)
for i in np.arange(Unique_Orig_Sample_IDs.shape[0]):
    Orig_Inds = np.where(Orig_Sample_IDs == Unique_Orig_Sample_IDs[i])[0]
    Predictions = Gao2019_sc_Predictions[Orig_Inds]
    Unique_Predictions = np.unique(Predictions,return_counts=True)
    Unique_Prediction_Counts = Unique_Predictions[1]
    Unique_Predictions = Unique_Predictions[0]
    for j in np.arange(Unique_Predictions.shape[0]):
        Inds = np.where(Predictions == Unique_Predictions[j])[0]
        Ratio = Inds.shape[0] / Predictions.shape[0]
        Gao2019_Ratios[Unique_Predictions[j]][i] = Ratio

Gao2019_Ratios.index = np.array(["Gao 2019\n8-cell (EPSCs)"])


Orig_Sample_IDs = Mazid2022_Sample_Labels.copy()
Unique_Orig_Sample_IDs = np.unique(Orig_Sample_IDs)
Mazid2022_Ratios = pd.DataFrame(np.zeros((Unique_Orig_Sample_IDs.shape[0],Unique_Manual_Annotations.shape[0])),columns=Unique_Manual_Annotations)
for i in np.arange(Unique_Orig_Sample_IDs.shape[0]):
    Orig_Inds = np.where(Orig_Sample_IDs == Unique_Orig_Sample_IDs[i])[0]
    Predictions = Mazid2022_sc_Predictions[Orig_Inds]
    Unique_Predictions = np.unique(Predictions,return_counts=True)
    Unique_Prediction_Counts = Unique_Predictions[1]
    Unique_Predictions = Unique_Predictions[0]
    for j in np.arange(Unique_Predictions.shape[0]):
        Inds = np.where(Predictions == Unique_Predictions[j])[0]
        Ratio = Inds.shape[0] / Predictions.shape[0]
        Mazid2022_Ratios[Unique_Predictions[j]][i] = Ratio

Mazid2022_Ratios.index = np.array(["Mazid 2022\nNaive (4CL)","Mazid 2022\n8-cell like","Mazid 2022\nPrimed (mTeSR 1)","Mazid 2022\nNaive (e4CL)"])


Combined_Ratios = pd.concat([Messmer2019_Ratios,Guo2021_sc_Ratios,Gao2019_Ratios,Mazid2022_Ratios])
Plot_Cell_Types = np.array(['preIm-Epi','Embryonic Disc','8-Cell'])#,'ICM/TE Branch'])
Combined_Ratios = Combined_Ratios[Plot_Cell_Types]
Combined_Ratios["Other"] = 1 - np.sum(Combined_Ratios,axis=1)
Combined_Ratios = Combined_Ratios.loc[["Guo 2021\nNaive (PGXL)","Messmer 2019\nNaive (t2iLG)","Mazid 2022\nNaive (e4CL)","Mazid 2022\nNaive (4CL)","Messmer 2019\nPrimed (E8)","Gao 2019\n8-cell (EPSCs)","Mazid 2022\nPrimed (mTeSR 1)"]]

Use_Colours = []
for i in np.arange(Plot_Cell_Types.shape[0]):
    Label = Plot_Cell_Types[i]
    Use_Colours.append(np.array(Colours)[np.where(np.isin(Unique_Manual_Annotations,Label))[0]])

Use_Colours = np.append(Use_Colours,"grey")


ax = Combined_Ratios.plot(kind='bar', stacked=True, color=Use_Colours,legend=False,figsize=(6,4.5),edgecolor = "black",)
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.xlabel("Dataset and original annotation",fontsize=12,)
plt.ylabel("Human embryo classifier predictions",fontsize=11)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
ax.set_xticklabels(ax.get_xticklabels(),fontweight='bold')
Label_Colours = np.asarray(["tab:cyan","tab:olive","tab:green", "tab:blue", "tab:pink", "tab:purple","tab:orange"])
for i in np.arange(Label_Colours.shape[0]):
    plt.gca().get_xticklabels()[i].set_color(Label_Colours[i])

plt.tight_layout()
plt.savefig(human_embryo_path + "Plots/" + "Prediction_Ratios.png",dpi=800)
plt.close()
plt.show()

####


Embryo_Naive = np.mean(Human_Embryo_Counts.iloc[np.where(Human_Sample_Info["Manual_Annotations"] == 'preIm-Epi')[0]][Marker_Genes],axis=0)
Embryo_Primed = np.mean(Sub_Human_Embryo_Counts.iloc[np.where(Human_Sample_Info["Manual_Annotations"] == 'Embryonic Disc')[0]][Marker_Genes],axis=0)

Guo2021_Naive = np.mean(Sub_Guo2021_sc_Counts.iloc[np.where(Guo2021_sc_Predictions == 'preIm-Epi')[0]][Marker_Genes],axis=0)
Messmer2019_Naive = np.mean(Sub_Messmer2019_Counts.iloc[np.where(Messmer2019_sc_Predictions == 'preIm-Epi')[0]][Marker_Genes],axis=0)
Mazid2022_e4CL_Naive = np.mean(Sub_Mazid2022_Counts.iloc[np.where((Mazid2022_sc_Predictions == 'Embryonic Disc') & (Mazid2022_Sample_Labels == "e4CL"))[0]][Marker_Genes],axis=0)
Mazid_2022_4CL_Primed = np.mean(Sub_Mazid2022_Counts.iloc[np.where((Mazid2022_sc_Predictions == 'Embryonic Disc') & (Mazid2022_Sample_Labels == "4CL"))[0]][Marker_Genes],axis=0)
Messmer2019_Primed = np.mean(Sub_Messmer2019_Counts.iloc[np.where(Messmer2019_sc_Predictions == 'Embryonic Disc')[0]][Marker_Genes],axis=0)
Mazid_2022_Primed = np.mean(Sub_Mazid2022_Counts.iloc[np.where(Mazid2022_sc_Predictions == 'Embryonic Disc')[0]][Marker_Genes],axis=0)
Gao2019_Primed = np.mean(Sub_Gao2019_Counts.iloc[np.where(Gao2019_sc_Predictions == 'Embryonic Disc')[0]][Marker_Genes],axis=0)

Combined_Exp_Data = pd.concat([Embryo_Naive,Guo2021_Naive,Messmer2019_Naive,Mazid2022_e4CL_Naive,Mazid_2022_4CL_Primed,Messmer2019_Primed,Embryo_Primed,Gao2019_Primed,Mazid_2022_Primed],axis=1)#,Embryo_8cell,Mazid2022_8cell,Embryo_ICM_TE,Mazid2022_ICM_TE],axis=1)
x_labels = np.array(["Embryo\npreIm-Epi","Guo 2021\nNaive (PGXL)","Messmer 2019\nNaive (t2iLG)","Mazid 2022\nNaive (e4CL)","Mazid 2022\nNaive (4CL)","Embryo\nEmbryonic Disc","Messmer 2019\nPrimed (E8)","Gao 2019\n8-cell (EPSCs)","Mazid 2022\nPrimed (mTeSR 1)"])#,"Embryo_8cell","Mazid2022_8cell","Embryo_ICM_TE","Mazid2022_ICM_TE"])
Combined_Exp_Data.columns = x_labels

Combined_Exp_Data = Combined_Exp_Data.apply(zscore)
Combined_Exp_Data = Combined_Exp_Data.T.apply(zscore)

g = sns.clustermap(Combined_Exp_Data.T,metric="correlation",cmap="seismic",col_cluster=False,figsize=(6,4.5))
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=70)
col = g.ax_col_dendrogram.get_position()
g.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*0.25])
row = g.ax_row_dendrogram.get_position()
g.ax_row_dendrogram.set_position([row.x0+0.12, row.y0, row.width*0.25, row.height*1])
Label_Colours = np.asarray(["#DA16FF","tab:cyan","tab:olive","tab:green","tab:blue","#B68100","tab:pink", "tab:purple","tab:orange"])
x_labels
for i in np.arange(Label_Colours.shape[0]):
    g.ax_heatmap.axes.get_xticklabels()[i].set_color(Label_Colours[i])

g.ax_heatmap.axes.set_xticklabels(g.ax_heatmap.axes.get_xticklabels(),fontweight='bold')
plt.savefig(human_embryo_path + "Plots/" + "Naive_Prime_Heatmap.png",dpi=800)
plt.close()
plt.show()


plt.show()


#################














Spacing = np.array([13706,11000,9000,7000,5000,4000,3000,2000,1000])


for i in np.arange(Spacing.shape[0]):
    Use_Inds = np.argsort(-Normalised_Network_Feature_Weights)[np.arange(Spacing[i])] 
    Selected_Genes = Subset_Used_Features[Use_Inds]
    #
    Neighbours = 30
    Dist = 0.1
    Gene_Embedding = umap.UMAP(n_neighbors=Neighbours, min_dist=Dist, n_components=2).fit_transform(Masked_ESSs[np.ix_(Use_Inds,Use_Inds)])
    #
    plt.figure(figsize=(2.33,2.33))
    plt.title("Top " + str(Spacing[i]) + " cESFW genes", fontsize=12)
    plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=0.1,c=np.sum(Masked_ESSs[np.ix_(Use_Inds,Use_Inds)] > 0,axis=0))
    #plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(0.02,0.02,0.98,0.9)
    plt.savefig(human_embryo_path + "Top " + str(Spacing[i]) + " cESFW genes.png",dpi=600)
    plt.close()


plt.show()





import gseapy as gp

Human = gp.get_library_name(organism='Human')
#go_mf = gp.get_library(name='GO_Molecular_Function_2018', organism='Yeast')

Chosen_Genes = Selected_Genes[np.isin(Selected_Genes,Saved_cESFW_Genes)].tolist()
Other_Genes = Selected_Genes[np.isin(Selected_Genes,Saved_cESFW_Genes)==0].tolist()


enr_1 = gp.enrichr(gene_list=Chosen_Genes, # or "./tests/data/gene_list.txt",
                 gene_sets=["GO_Biological_Process_2021"],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                 background=Human_Embryo_Counts.columns.tolist(),
                )



enr_2 = gp.enrichr(gene_list=Other_Genes, # or "./tests/data/gene_list.txt",
                 gene_sets=["GO_Biological_Process_2021"],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                 background=Human_Embryo_Counts.columns.tolist(),
                )




enr.results.Term = enr.res2d.Term.str.split(" \(GO").str[0]
Terms = enr.results["Term"][np.arange(30)]
Values = enr.results["Adjusted P-value"][np.arange(30)]
#Values = np.log10(Values)

#plt.figure(figsize=(5,5))
plt.barh(Terms, -Values)
plt.tight_layout()


enr_2.results.Term = enr_2.res2d.Term.str.split(" \(GO").str[0]
Terms = enr_2.results["Term"][np.arange(30)]
Values = enr_2.results["Adjusted P-value"][np.arange(30)]
#Values = np.log10(Values)

#plt.figure(figsize=(5,5))
plt.barh(Terms, -Values)
plt.tight_layout()
plt.show()





# 8-cell: ZSCAN4, TPRX1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8901440/

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("ZSCAN4 (8-cell)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["ZSCAN4"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "ZSCAN4.png",dpi=800)
plt.close()


np.isin("DPRX",Saved_cESFW_Genes)


# Morula: DPRX, BTN1A1. https://pubmed.ncbi.nlm.nih.gov/29361568/

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("BTN1A1 (Morula)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["BTN1A1"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "BTN1A1.png",dpi=800)
plt.close()


# ICM/TE branch: BAIAP2L2, https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002162, https://journals.biologists.com/dev/article/145/3/dev158501/19263/Integrated-analysis-of-single-cell-embryo-data

np.isin("BAIAP2L2",Saved_cESFW_Genes)

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("BAIAP2L2 (ICM/TE branch)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["BAIAP2L2"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "BAIAP2L2.png",dpi=800)
plt.close()


# Early Hypoblast: LGALS2. https://doi.org/10.1242/dev.201522

np.isin("LGALS2",Saved_cESFW_Genes)

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("LGALS2 (Early hypoblast)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["LGALS2"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "LGALS2.png",dpi=800)
plt.close()

# Hypoblast: FOXA2. https://doi.org/10.1242/dev.201522

np.isin("FLRT3",Saved_cESFW_Genes)

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("FLRT3 (Hypoblast)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["FLRT3"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "FLRT3.png",dpi=800)
plt.close()


# Early TE: CYP26A1, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9365277/

np.isin("CYP26A1",Saved_cESFW_Genes)

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("CYP26A1 (Early TE)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["CYP26A1"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "CYP26A1.png",dpi=800)
plt.close()


# Mid TE: CYP26A1, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5671803/

np.isin("RGS13",Saved_cESFW_Genes)

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("RGS13 (Mid TE)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["RGS13"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "RGS13.png",dpi=800)
plt.close()


# Mural TE: HAND1, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9365277/

np.isin("SERPINC1",Saved_cESFW_Genes) # This gene is in set, but can't find paper.

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("HAND1 (Mural TE)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["HAND1"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "HAND1.png",dpi=800)
plt.close()


# Polar TE: CCR7, CYP19A1, DLX5, MUC15. https://www.science.org/doi/10.1126/sciadv.abj3725
# Polar TE: GAST. https://www.fertstert.org/article/S0015-0282(20)30009-1/fulltext

np.isin("GAST",Saved_cESFW_Genes)

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("GAST (Polar TE)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["GAST"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "GAST.png",dpi=800)
plt.close()


# Syncytiotrophoblast: SLC38A3, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4868474/

np.isin("SLC38A3",Saved_cESFW_Genes) 

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("SLC38A3 (Syncytiotrophoblast)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["SLC38A3"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "SLC38A3.png",dpi=800)
plt.close()


# Cytotrophoblast : ISM2, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6686486/

np.isin("ISM2",Saved_cESFW_Genes) 

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("ISM2 (Cytotrophoblast)",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["ISM2"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "ISM2.png",dpi=800)
plt.close()


# Extraembryonic mesenchyme : HGF, https://www.nature.com/articles/s41467-021-25186-2

np.isin("HGF",Saved_cESFW_Genes) 

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("HGF (extraembryonic mesenchyme)",fontsize=9)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["HGF"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "HGF.png",dpi=800)
plt.close()

Annotation_ESSs.index[np.argsort(-Annotation_ESSs['Putative Amn'])][np.arange(30)]




fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("TBX20 (extraembryonic mesenchyme)",fontsize=8)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["TBX20"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "TBX20.png",dpi=800)
plt.close()


Gene = "ESRRB"

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title(Gene ,fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts[Gene]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + Gene +".png",dpi=800)
plt.close()








# Manu genes: "CASP6","CASP7","CTSB","CYP26A1","TUBA3C","BAK1","BIK","ATF3","ATG2A","PRKCQ"
# OLAH, CYP26A1, ETNPPL, FAM163B, REM1, MAG


Manu_NCC_IDs = np.load(human_embryo_path+"Manu_NCC_IDs.npy")
Manu_NCC_Inds = np.where(np.isin(Human_Sample_Info.index,Manu_NCC_IDs))[0]
#
Manu_ICM_IDs = np.load(human_embryo_path+"Manu_ICM_IDs.npy")
Manu_ICM_Inds = np.where(np.isin(Human_Sample_Info.index,Manu_ICM_IDs))[0]
#
Manu_TE_IDs = np.load(human_embryo_path+"Manu_TE_IDs.npy")
Manu_TE_Inds = np.where(np.isin(Human_Sample_Info.index,Manu_TE_IDs))[0]

plt.figure(figsize=(8,8))
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=4,c="grey")
plt.scatter(Human_Embryo_Embedding[Manu_NCC_Inds,0],Human_Embryo_Embedding[Manu_NCC_Inds,1],s=8)
plt.scatter(Human_Embryo_Embedding[Manu_ICM_Inds,0],Human_Embryo_Embedding[Manu_ICM_Inds,1],s=8)
plt.scatter(Human_Embryo_Embedding[Manu_TE_Inds,0],Human_Embryo_Embedding[Manu_TE_Inds,1],s=8)

plt.show()


plt.figure(figsize=(2.33,2.33))
plt.title("Singh et al. 2023",fontsize=10)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey")
plt.scatter(Human_Embryo_Embedding[Manu_NCC_Inds,0],Human_Embryo_Embedding[Manu_NCC_Inds,1],s=0.5,label="Non characterised cells (NCCs)",c=Colours[17])
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Singh_NCCs.png",dpi=800)
plt.close()




Petrop_Sample_Info = pd.read_csv(human_embryo_path+"E-MTAB-3929.sdrf.txt",header=0,index_col=0,sep="\t")


Petrop_Sample_Info["Characteristics[inferred lineage]"]
Petrop_Sample_Info["Characteristics[inferred trophectoderm subpopulation]"]

np.unique(Petrop_Sample_Info["Characteristics[treatment]"])

Petrop_Sample_IDs = Petrop_Sample_Info.index
Human_Sample_IDs = Human_Sample_Info.index

Immunosurgary = np.repeat("                   nan                 ",Sample_IDs.shape[0])
Polar_Mural_TE = np.repeat("                   nan                 ",Sample_IDs.shape[0])

for i in np.arange(Petrop_Sample_IDs.shape[0]):
    Ind = np.where(Human_Sample_IDs == Petrop_Sample_IDs[i])[0]
    Immunosurgary[Ind] = Petrop_Sample_Info["Characteristics[treatment]"][i]
    Polar_Mural_TE[Ind] = Petrop_Sample_Info["Characteristics[inferred trophectoderm subpopulation]"][i]


Polar_Mural_TE[np.where(Polar_Mural_TE == 'not applicable')[0]] = "                   nan                 "
Polar_Mural_TE[np.where(Polar_Mural_TE == "                   nan                 ")[0]] = "nan"
Polar_Mural_TE[np.where(Polar_Mural_TE == "mural")[0]] = "Mural TE"
Polar_Mural_TE[np.where(Polar_Mural_TE == "polar")[0]] = "Polar TE"
Immunosurgary[np.where(Immunosurgary == "Immunos")[0]] = "Immunosurgery"
Immunosurgary[np.where(Immunosurgary == "No")[0]] = "No immunosurgery"
Immunosurgary[np.where(Immunosurgary == "                   nan                 ")[0]] = "nan"

plt.figure(figsize=(2.33,2.33))
plt.title("Petropoulos et al. 2016",fontsize=10)
Unique_Polar_Mural_TE = np.unique(Polar_Mural_TE)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey")
IDs = np.where(Polar_Mural_TE == 'Mural TE')[0]
plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c="tab:olive")
IDs = np.where(Polar_Mural_TE == 'Polar TE')[0]
plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c="tab:cyan")

plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Petropoulos_Polar_Mural.png",dpi=800)
plt.close()


plt.figure(figsize=(2.33,2.33))
plt.title("Petropoulos et al. 2016",fontsize=10)
Unique_Immunosurgary = np.unique(Immunosurgary)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.25,c="grey")
IDs = np.where(Immunosurgary == 'No immunosurgery')[0]
plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c="tab:cyan")
IDs = np.where(Immunosurgary == 'Immunosurgery')[0]
plt.scatter(Human_Embryo_Embedding[IDs,0],Human_Embryo_Embedding[IDs,1],s=0.5,c="tab:olive")


plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
#
plt.savefig(human_embryo_path + "Petropoulos_Immunosurgery.png",dpi=800)
plt.close()



















Used_Features = np.load(human_embryo_path + "Used_Features.npy",allow_pickle=True)

from scipy.stats import zscore

Unique_Manual_Annotations = np.unique(Manual_Annotations)

Annotation_Labels = np.zeros(Human_Sample_Info.shape[0])
Annotation_Labels[np.where(np.isin(Manual_Annotations,np.array(['Epi/Hypo Branch', 'preIm-Epi'])))[0]] = 1
#Annotation_Labels[np.where(np.isin(Manual_Annotations,np.array(["Morula","ICM/TE Branch","ICM","Early-Hypo","preIm-Epi"])))[0]] = 1
# If the number of samples in the dataset is greater than half the cardinality of the entire data, flip the majority/minority state labels.
if np.sum(Annotation_Labels) > (Human_Sample_Info.shape[0]/2):
    Annotation_Labels = (Annotation_Labels * - 1) + 1
## Calulate the ESSs and EPs of the dataset with every feature/gene in the scaled matrix.
Individual_ESSs, Individual_EPs = cESFW.Calculate_Individual_ESS_EPs(Annotation_Labels,human_embryo_path)


#Annotation_ESSs.to_csv(human_embryo_path+"Annotation_ESSs.csv")


Plot_Genes = Used_Features[np.argsort(-Individual_ESSs)][np.arange(30)]

for i in np.arange(Plot_Genes.shape[0]):
    plt.figure(figsize=(10,10))
    plt.title(Plot_Genes[i], fontsize=20)
    plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=8,c=np.log2(Human_Embryo_Counts[Plot_Genes[i]]+1),cmap="seismic")

plt.show()

# "SLC25A12","SOX2","MT1X","MT1G","FAM151A","ARGFX"

Epiblast_Emergence_Genes = np.array(["CORO1A","ACE","SLC47A1","DEFB124", "GPR160", # Morula to preImp-Epi
                                     "GDPD2","RRAD","ITGAM","GPR176", "GK","PI16", # ICM/TE branch to preImp-Epi
                                     "MAN1C1","CCR8","ANKRD45","CDHR1","MT1H", # ICM to preImp-Epi
                                     "KLK13","FBP1","C9orf135","KLHL3","APOBEC3C","ETV1","VENTX","UTF1",# Epi/Hypo to preImp-Epi
                                      "CD70","LEFTY1","HRK","ARX","ATP12A","LEFTY2"]) # preImp-Epi



Selected_Genes_Expression = Human_Embryo_Counts[Epiblast_Emergence_Genes]
Selected_Genes_Expression = np.log2(Selected_Genes_Expression+1)


Pseudobulk_Selected_Genes_Expression = pd.DataFrame(np.zeros((Unique_Manual_Annotations.shape[0],Epiblast_Emergence_Genes.shape[0])),columns=Epiblast_Emergence_Genes,index=Unique_Manual_Annotations)

for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    Expression = Selected_Genes_Expression.iloc[IDs]
    Pseudobulk_Selected_Genes_Expression.iloc[i] = np.mean(Expression,axis=0)

Pseudobulk_Selected_Genes_Expression = Pseudobulk_Selected_Genes_Expression.apply(zscore)

# Pseudobulk_Selected_Genes_Expression = Pseudobulk_Selected_Genes_Expression / np.max(Epiblast_Emergence_Genes,axis=0)

Selected_Timepoints = np.array(['8-Cell', 'Morula', 'ICM/TE Branch', 'ICM', 'Epi/Hypo Branch', 'preIm-Epi', 'Embryonic Disc', 'ExE-Mech', 'Hypo', 'Early TE', 'Mid TE', 'Mural TE', 'Polar TE', 'cTB', 'sTB'])
# Selected_Timepoints = np.append(Selected_Timepoints,Unique_Manual_Annotations[np.isin(Unique_Manual_Annotations,Selected_Timepoints)==0])
# Selected_Timepoints = np.delete(Selected_Timepoints,np.where(Selected_Timepoints=="Unknown")[0])

# xlabels = Selected_Genes.copy()
# xlabels[np.isin(xlabels,Human_TFs)==0] = ""

cm = sns.clustermap(Pseudobulk_Selected_Genes_Expression.loc[Selected_Timepoints],cmap="seismic",row_cluster=False,col_cluster=False,xticklabels=True,figsize=(9,5),cbar_pos=None)
# plt.title("Epiblast emergence",fontsize=16)
plt.setp(cm.ax_heatmap.get_xticklabels(), rotation=70)
plt.tight_layout()
# plt.savefig(human_embryo_path + "Plots/" + "Epiblast_emergence.png",dpi=800)
# plt.close()
plt.show()





Annotation_Labels = np.zeros(Human_Sample_Info.shape[0])
Annotation_Labels[np.where(np.isin(Manual_Annotations,np.array(['Mural TE', 'Polar TE'])))[0]] = 1
#Annotation_Labels[np.where(np.isin(Manual_Annotations,np.array(["Morula","ICM/TE Branch","ICM","Early-Hypo","preIm-Epi"])))[0]] = 1
# If the number of samples in the dataset is greater than half the cardinality of the entire data, flip the majority/minority state labels.
# if np.sum(Annotation_Labels) > (Human_Sample_Info.shape[0]/2):
#     Annotation_Labels = (Annotation_Labels * - 1) + 1
## Calulate the ESSs and EPs of the dataset with every feature/gene in the scaled matrix.
Individual_ESSs, Individual_EPs = cESFW.Calculate_Individual_ESS_EPs(Annotation_Labels,human_embryo_path)


#Annotation_ESSs.to_csv(human_embryo_path+"Annotation_ESSs.csv")


Plot_Genes = Used_Features[np.argsort(-Individual_ESSs)][np.arange(30)]

for i in np.arange(Plot_Genes.shape[0]):
    plt.figure(figsize=(10,10))
    plt.title(Plot_Genes[i], fontsize=20)
    plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=8,c=np.log2(Human_Embryo_Counts[Plot_Genes[i]]+1),cmap="seismic")

plt.show()


TE_Emergence_Genes = np.array(["SLC28A3", "ECE2" ,"Z82206.1", "ZNF593", "GNG10", # Morula to TE
                                     "ATP6V0A4", "ATP6V1B1", "ALPP", "RAB25", # ICM/TE branch to TE
                                     "SLC7A4", "ENPEP", "ANXA8", "TACSTD2", "EMP2","C4BPB", # Early TE to TE
                                     "HAPLN1", "CA12", "CLDN3", "TMEM52B", "TMPRSS13", "TRIML1", "ODAM","IRX2", # Mid TE to TE
                                     "TBX2","WNT6","PLEKHA6","TP63","RNF43","NR2F2"]) # TE



Selected_Genes_Expression = Human_Embryo_Counts[TE_Emergence_Genes]
Selected_Genes_Expression = np.log2(Selected_Genes_Expression+1)


Pseudobulk_Selected_Genes_Expression = pd.DataFrame(np.zeros((Unique_Manual_Annotations.shape[0],TE_Emergence_Genes.shape[0])),columns=TE_Emergence_Genes,index=Unique_Manual_Annotations)

for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    Expression = Selected_Genes_Expression.iloc[IDs]
    Pseudobulk_Selected_Genes_Expression.iloc[i] = np.mean(Expression,axis=0)

Pseudobulk_Selected_Genes_Expression = Pseudobulk_Selected_Genes_Expression.apply(zscore)

# Pseudobulk_Selected_Genes_Expression = Pseudobulk_Selected_Genes_Expression / np.max(TE_Emergence_Genes,axis=0)

Selected_Timepoints = np.array(['8-Cell', 'Morula', 'ICM/TE Branch', 'Early TE', 'Mid TE', 'Mural TE', 'Polar TE', 'cTB', 'sTB', 'ICM', 'Epi/Hypo Branch', 'preIm-Epi', 'Embryonic Disc', 'ExE-Mech', 'Hypo'])
# Selected_Timepoints = np.append(Selected_Timepoints,Unique_Manual_Annotations[np.isin(Unique_Manual_Annotations,Selected_Timepoints)==0])
# Selected_Timepoints = np.delete(Selected_Timepoints,np.where(Selected_Timepoints=="Unknown")[0])

# xlabels = Selected_Genes.copy()
# xlabels[np.isin(xlabels,Human_TFs)==0] = ""

cm = sns.clustermap(Pseudobulk_Selected_Genes_Expression.loc[Selected_Timepoints],cmap="seismic",row_cluster=False,col_cluster=False,xticklabels=True,figsize=(9,5),cbar_pos=None)
# plt.title("Trophectoderm emergence",fontsize=16)
plt.setp(cm.ax_heatmap.get_xticklabels(), rotation=70)
plt.tight_layout()
# plt.savefig(human_embryo_path + "Plots/" + "TE_emergence.png",dpi=800)
# plt.close()
plt.show()







Annotation_Labels = np.zeros(Human_Sample_Info.shape[0])
Annotation_Labels[np.where(np.isin(Manual_Annotations,np.array(['ICM/TE Branch', 'ICM', 'Epi/Hypo Branch', 'Hypo'])))[0]] = 1
#Annotation_Labels[np.where(np.isin(Manual_Annotations,np.array(["Morula","ICM/TE Branch","ICM","Early-Hypo","preIm-Epi"])))[0]] = 1
# If the number of samples in the dataset is greater than half the cardinality of the entire data, flip the majority/minority state labels.
# if np.sum(Annotation_Labels) > (Human_Sample_Info.shape[0]/2):
#     Annotation_Labels = (Annotation_Labels * - 1) + 1
## Calulate the ESSs and EPs of the dataset with every feature/gene in the scaled matrix.
Individual_ESSs, Individual_EPs = cESFW.Calculate_Individual_ESS_EPs(Annotation_Labels,human_embryo_path)


#Annotation_ESSs.to_csv(human_embryo_path+"Annotation_ESSs.csv")


Plot_Genes = Used_Features[np.argsort(-Individual_ESSs)][np.arange(30)]

for i in np.arange(Plot_Genes.shape[0]):
    plt.figure(figsize=(10,10))
    plt.title(Plot_Genes[i], fontsize=20)
    plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=8,c=np.log2(Human_Embryo_Counts[Plot_Genes[i]]+1),cmap="seismic")

plt.show()

# "SOAT2", "SLC35E4"
Hypo_Emergence_Genes = np.array(["HNF4A", "ANKRD55", "EGFLAM","CLDN19","DHRS3", # Morula to Hypo
                                 "SPATS2L","RGS9", "NRGN","CEACAM6","CEACAM1", # ICM/TE branch to Hypo
                                 "PDGFRA", "BMP2","HNF1B","NRGN","SOX17","CTSE","LGALS2", # ICM to Hypo
                                 "GATA4","DPP4",# Epi/Hypo Branch to Hypo
                                 "FLRT3", "FRZB", "APOA1", "RSPO3", "FOXA2","CPN1"]) # Hypo



Selected_Genes_Expression = Human_Embryo_Counts[Hypo_Emergence_Genes]
Selected_Genes_Expression = np.log2(Selected_Genes_Expression+1)


Pseudobulk_Selected_Genes_Expression = pd.DataFrame(np.zeros((Unique_Manual_Annotations.shape[0],Hypo_Emergence_Genes.shape[0])),columns=Hypo_Emergence_Genes,index=Unique_Manual_Annotations)

for i in np.arange(Unique_Manual_Annotations.shape[0]):
    IDs = np.where(Manual_Annotations == Unique_Manual_Annotations[i])[0]
    Expression = Selected_Genes_Expression.iloc[IDs]
    Pseudobulk_Selected_Genes_Expression.iloc[i] = np.mean(Expression,axis=0)

Pseudobulk_Selected_Genes_Expression = Pseudobulk_Selected_Genes_Expression.apply(zscore)

# Pseudobulk_Selected_Genes_Expression = Pseudobulk_Selected_Genes_Expression / np.max(Hypo_Emergence_Genes,axis=0)

Selected_Timepoints = np.array(['8-Cell', 'Morula', 'ICM/TE Branch', 'ICM', 'Epi/Hypo Branch', 'Hypo', 'preIm-Epi', 'Embryonic Disc', 'ExE-Mech', 'Early TE', 'Mid TE', 'Mural TE', 'Polar TE', 'cTB', 'sTB'])
# Selected_Timepoints = np.append(Selected_Timepoints,Unique_Manual_Annotations[np.isin(Unique_Manual_Annotations,Selected_Timepoints)==0])
# Selected_Timepoints = np.delete(Selected_Timepoints,np.where(Selected_Timepoints=="Unknown")[0])

# xlabels = Selected_Genes.copy()
# xlabels[np.isin(xlabels,Human_TFs)==0] = ""


cm = sns.clustermap(Pseudobulk_Selected_Genes_Expression.loc[Selected_Timepoints],cmap="seismic",row_cluster=False,col_cluster=False,xticklabels=True,figsize=(9,5),cbar_pos=None)
# plt.title("Hypoblast emergence",fontsize=16)
plt.setp(cm.ax_heatmap.get_xticklabels(), rotation=70)
plt.tight_layout()
# plt.savefig(human_embryo_path + "Plots/" + "Hypoblast_emergence.png",dpi=800)
# plt.close()
plt.show()




# ACE, HNF4A, SLC28A3
Gene = "C4BPB"
plt.figure(figsize=(10,10))
plt.title(Gene, fontsize=20)
plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=8,c=np.log2(Human_Embryo_Counts[Gene]+1),cmap="seismic")

plt.show()

fig, ax = plt.subplots(figsize=(2.33,2.33))
plt.title("SLC47A1",fontsize=10)
cmap_plot = plt.scatter(Human_Embryo_Embedding[:,0],Human_Embryo_Embedding[:,1],s=0.5,c=np.log2(Human_Embryo_Counts["SLC47A1"]+1),cmap="seismic")
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.82,0.52,0.05,0.42)))
cb.set_label('$log_2$(Expression)', labelpad=-31,fontsize=7)
cb.ax.tick_params(labelsize=7)
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(human_embryo_path + "SLC47A1.png",dpi=800)
plt.close()

