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

	#turn lists into numpy arrays
	data_kmeans = data_list
	data_list = np.array(data_list)
	sampling_rate_list = np.array(sampling_rate_list)

	#run DBSCAN on our data
	clustering = DBSCAN(eps=68, min_samples=2).fit(data_list)
	clustercount = np.max(clustering.labels_) + 1
	clusteringlist = list(clustering.labels_)

	#Setup for outputting to file
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

	#writes DBSCAN output into output_DBSCAN.txt
	output_file = 'output/output_DBSCAN.txt'
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
		km = K_Means(k)
		labels = km.fit(data_kmeans)
		j += 1
	
	clusteringlist = labels
	
	#Setup for outputting to file
	zipped = zip(clusteringlist, all_files)
	zipped = set(zipped)

	num_clusters = [[] for i in range(1, clustercount+1)]
	
	for k,v in zipped:
		for i in range(len(zipped)):
			if k == i:
				num_clusters[i].append(v)
		i += 1

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
	
	#writes KMEANS output into output_KMEANS.txt
	output_file = 'output/output_KMEANS.txt'
	with open(output_file, 'w') as fw:
		fw.write('Number of Clusters: ' + str(len(list_to_output)) + '\n')
		i = 0
		for cluster in list_to_output:
			fw.write('Cluster ' + str(i) + ' contains: ' + str(cluster) + '\n')
			i += 1

	#Our own kmeans results
	# print("Our Own KMeans Results:")
	# for key, value in (clusters[1]).items():
	# 	print("Cluster {} contains ".format(key + 1) + str(len(value)) + " files")

	print('\n')

	#scikit agglomerative output
	print("Scikit Agglomerative Results:")
	clusters = sci_kit_agg_clustering(data_list, len(list_to_output))

	#Setup for outputting to file
	zipped = zip(clusters, all_files)
	zipped = set(zipped)

	num_clusters = [[] for i in range(1, clustercount+1)]
	
	for k,v in zipped:
		for i in range(len(zipped)):
			if k == i:
				num_clusters[i].append(v)
		i += 1
	
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
	
	#writes SCIKITAGG output into output_SCIKITAGG.txt
	output_file = 'output/output_SCIKITAGG.txt'
	with open(output_file, 'w') as fw:
		fw.write('Number of Clusters: ' + str(len(list_to_output)) + '\n')
		i = 0
		for cluster in list_to_output:
			fw.write('Cluster ' + str(i) + ' contains: ' + str(cluster) + '\n')
			i += 1

	print('\n')

	#scikit kmeans output
	print("Scikit KMeans Results:")
	clusters = sci_kit_KMeans(data_list, len(list_to_output))

	#Setup for outputting to file
	clusteringlist = clusters

	zipped = zip(clusteringlist, all_files)
	zipped = set(zipped)

	num_clusters = [[] for i in range(1, clustercount+1)]
	
	for k,v in zipped:
		for i in range(len(zipped)):
			if k == i:
				num_clusters[i].append(v)
		i += 1
	
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
	
	#writes SCIKITKMEANS output into output_SCIKITKMEANS.txt
	output_file = 'output/output_SCIKITKMEANS.txt'
	with open(output_file, 'w') as fw:
		fw.write('Number of Clusters: ' + str(len(list_to_output)) + '\n')
		i = 0
		for cluster in list_to_output:
			fw.write('Cluster ' + str(i) + ' contains: ' + str(cluster) + '\n')
			i += 1

#Scikit Agglomerative Complete-Link Clustering
def sci_kit_agg_clustering(vectors, k):
	x = np.array(vectors)
	agg_clustering = AgglomerativeClustering(n_clusters=k, linkage="complete").fit(x)
	labels = agg_clustering.labels_
	my_labels = list(dict.fromkeys(labels))
	my_labels = list(my_labels)
	cluster_totals = []
	i = 0
	return labels

#Scikit KMeans Clustering
def sci_kit_KMeans(vectors, k):
		x = np.array(vectors)
		k_means = KMeans(n_clusters=k).fit(x)
		labels = k_means.labels_
		print("labels is here", labels)
		my_labels = list(dict.fromkeys(labels))
		my_labels = list(my_labels)
		cluster_totals = []
		i = 0
		return labels


#Our own kmeans implementation
class K_Means:										#adapted from: https://pythonprogramming.net/k-means-from-scratch-2-machine-learning-tutorial/?completed=/k-means-from-scratch-machine-learning-tutorial/
													#fixed by Alex and Andrew
	def __init__(self, k=3, tol=0.1, max_iter=300):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter

	def fit(self,data):

		self.centroids = {}

		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for i in range(self.k):
				self.classifications[i] = []

			sum_distances = 0
			data_list = []
			for featureset in data:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
				for i in range(len(distances)):
					distances[i] = int(distances[i])
				classification = distances.index(min(distances))
				sum_distances += min(distances)
				data_list.append(classification)
				self.classifications[classification].append(featureset)

		return data_list

if __name__ == "__main__":
	main()