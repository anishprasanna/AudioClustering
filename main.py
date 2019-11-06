import glob
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import librosa.display
import librosa.feature

def main():
    #get rid of scientific notation in numpy arrays
    np.set_printoptions(suppress=True)
    features = 8

    all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
    all_files.sort()
    data_list = []
    sampling_rate_list = []

    #write feature set from every file to csv
    with open('features.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        for file in all_files:
            data, sr = librosa.load(file)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=features)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            data_list.append(mfccsscaled)
            writer.writerow(mfccsscaled)
    csvFile.close()

    # print("Pre numpy array data list: ", data_list)

    #turn lists into numpy arrays
    data_list = np.array(data_list)
    sampling_rate_list = np.array(sampling_rate_list)

    clustering = DBSCAN(eps=75, min_samples=2).fit(data_list)
    clustercount = np.max(clustering.labels_) + 1
    clusteringlist = list(clustering.labels_)

    zipped = zip(clusteringlist, all_files)
    zipped = set(zipped)

    num_clusters = [[] for i in range(1, clustercount+1)]
    
    for k,v in zipped:
        for i in range(len(zipped)):
            if k == i:
                num_clusters[i].append(v)
        i += 1
    
    #lines 75 - 97 written by Carlos S.

    #have to filter out each name and put it into respective sublists within big list
    sub_list_to_output = []
    list_to_output = []
    for item in num_clusters:
        for subitem in item:
            subitem = subitem.replace('assignment5/data/', '')
            subitem = subitem.replace('.wav', '')
            sub_list_to_output.append(subitem)
        list_to_output.append(sub_list_to_output)
        sub_list_to_output = []

    print('list to output in output file', list_to_output)

    #writes output into output.txt
    output_file = 'output/output.txt'
    with open(output_file, 'w') as fw:
        fw.write('Number of Clusters: ' + str(len(list_to_output)) + '\n')
        i = 0
        for cluster in list_to_output:
            fw.write('Cluster ' + str(i) + ' contains: ' + str(cluster) + '\n')
            i += 1
    
    #scikit agglomerative
    print("Scikit Agglomerative Results:")
    sci_kit_agg_clustering(data_list, len(list_to_output))

    print('\n')

    #scikit kmeans
    print("Scikit KMeans Results:")
    sci_kit_KMeans(data_list, len(list_to_output))


def sci_kit_agg_clustering(vectors, k):
    x = np.array(vectors)
    agg_clustering = AgglomerativeClustering(n_clusters=k, linkage="complete").fit(x)
    labels = agg_clustering.labels_
    my_labels = list(dict.fromkeys(labels))
    my_labels = list(my_labels)
    cluster_totals = []
    i = 0
    while i < len(my_labels):
        count = 0
        for item in labels:
            if item == my_labels[i]:
                count += 1
        cluster_totals.append(count)
        print("Total files in Cluster " + str(i+1) + ": " + str(count))
        i += 1

def sci_kit_KMeans(vectors, k):
        x = np.array(vectors)
        k_means = KMeans(n_clusters=k).fit(x)
        labels = k_means.labels_
        my_labels = list(dict.fromkeys(labels))
        my_labels = list(my_labels)
        cluster_totals = []
        i = 0
        while i < len(my_labels):
            count = 0
            for item in labels:
                if item == my_labels[i]:
                    count += 1
            cluster_totals.append(count)
            print("Total files in Cluster " + str(i+1) + ": " + str(count))
            i += 1

if __name__ == "__main__":
    main()