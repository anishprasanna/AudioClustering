# MFC Clustering - Alex Brockman, Andrew Haisfield, Anish Prasanna and Carlos Samaniego

In this project, we performed cluster analysis of 930 (.wav) files using their MFC (mel-frequency cepstral) coefficients generated through the LibROSA library. We used the coefficients as features for our selected algorithms optimized by DBSCAN. 

When running the code, you will see a simple output containing the compilation times for all of the algorithms included in our program. They include: Scikit's DBSCAN, KMeans++, agglomerative hierarchical clustering, and our own K-means implementation algorithm. The output files include a features list computed from the .wav files, a description of the location of the files in the clusters and an excel file that displays a concordance table for all the files in our data set. Also included is a report discussing the findings.

