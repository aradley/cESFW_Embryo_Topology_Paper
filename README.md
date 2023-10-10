# cESFW_Embryo_Topology_Paper
In this repository we provide the code to reproduce the results in our manuscript (add hyperlink here).

## cESFW software package
The research in our manuscript focusses on introducing our continuous entropy sort feature weighting (cESFW) software. This package can be downloaded and installed from https://github.com/aradley/cESFW

## Data repository
The synthetic data and human embryo data required to reproduce the results in our manuscript and run the workflow provided in the following Mendeley Data repository can be found at https://data.mendeley.com/datasets/34td4ds2r9/draft?a=c464ae1c-08a6-430a-8bb2-6dc8146d61f5

## Synthetic data workflows
Each of the 4 synthetic datasets used in our manuscript can be reproduced by going to the synthetic dataset folder of interest and following the data generation workflows which are written in R. For example, after downloading the above Mendeley Data repository, navigate to the Synthetic_Data folder and then select one of the 4 sub-directories. Within these, the file name starting with "Dyngen_Create" will allow the user to create the synthetic dataset from scratch in R, and the file name ending with "workflow" will allow the user to analyse the data in Python.

## Human embryo scRNA-seq workflow
### cESFW_Feature_Selection_Workflow
The cESFW_Feature_Selection_Workflow workflow in this repository walks the user through the cESFW workflow used in our mansucript to select a subset of highly informative genes from the provideded human embryo scRNA-seq counts matrix. 
### Human_Embryo_Plotting_Workflow
The cESFW_Feature_Selection_Workflow workflow in this repository allows the user to re-create all of the plots presented in our manuscript.









