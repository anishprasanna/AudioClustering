import glob
import os
import csv
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import librosa.display
import librosa.feature

def main():
    #get rid of scientific notation in numpy arrays
    np.set_printoptions(suppress=True)
    features = 40

    all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
    all_files.sort()
    data_list = []
    sampling_rate_list = []

    #Writing feature sets to csv written by Alex B.

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

    #Optimal EPS value code written by Alex B.

    # #find the optimal eps value
    # neigh = NearestNeighbors(n_neighbors=4)
    # nbrs = neigh.fit(data_list)
    # distances, indices = nbrs.kneighbors(data_list)

    # distances = np.sort(distances, axis=0)
    # distances = distances[:,1]
    # plt.plot(distances)
    # plt.show()



    #print("Pre numpy array data list: ", data_list)

    #turn lists into numpy arrays
    data_list = np.array(data_list)
    sampling_rate_list = np.array(sampling_rate_list)

    clustering = DBSCAN(eps=68, min_samples=2).fit(data_list)
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

    #print('list to output in output file', list_to_output)

    #writes output into output.txt
    output_file = 'output/output.txt'
    with open(output_file, 'w') as fw:
        fw.write('Number of Clusters: ' + str(len(list_to_output)) + '\n')
        i = 0
        for cluster in list_to_output:
            fw.write('Cluster ' + str(i) + ' contains: ' + str(cluster) + '\n')
            i += 1
    
    #Outputting our own KMeans
    j = 0
    while (j < 5):
        k = (len(list_to_output))
        centroids = {}
        for i in range(k):
            centroids[i] = random.choice(data_list)

        km = K_Means(k)

        clusters = km.fit(data_list, centroids)
        j += 1

    print("Our Own KMeans Results:")
    for key, value in (clusters[1]).items():
	    print("Cluster {} contains ".format(key + 1) + str(len(value)) + " files")

    print('\n')

    #scikit agglomerative output
    print("Scikit Agglomerative Results:")
    sci_kit_agg_clustering(data_list, len(list_to_output))

    print('\n')

    #scikit kmeans output
    print("Scikit KMeans Results:")
    sci_kit_KMeans(data_list, len(list_to_output))

#Scikit Agglomerative Complete-Link Clustering
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

#Scikit KMeans Clustering
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



class K_Means:		#Adapted from the URL: https://github.com/madhug-nadig/Machine-Learning-Algorithms-from-Scratch/blob/master/K%20Means%20Clustering.py, 
					#fixed by Alex and Andrew

#Fixed by Alex
	def __init__(self, k =3, tolerance = 0.0001, max_iterations = 500):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations

#.fit() function by Alex
	def fit(self, vectors, centroids):
		self.centroids = {}

		#begin iterations
		for i in range(self.max_iterations):
			self.classes = {}
			for i in range(self.k):
				self.classes[i] = []
			#print("classes", self.classes)

		#find the distance between the point and cluster; choose the nearest centroid
		sum_distances = 0
	#sum_of_distances by Andrew
		for features in vectors:
			distances = [np.linalg.norm(list(set(features) - set(centroids[centroid]))) for centroid in centroids]
			for i in range(len(distances)):
				distances[i] = int(distances[i])
			
			classification = distances.index(min(distances))
			sum_distances += min(distances)
			self.classes[classification].append(features)

		#print("Sum distances: " + str(sum_distances))
		return sum_distances, self.classes


if __name__ == "__main__":
    main()