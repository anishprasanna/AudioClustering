import glob
import os
import csv
import re
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
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

	# all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
	# print(all_files)

	# #Writing file names to csv written by Alex B.
	# with open('all_files.csv', 'w') as csvFile:
	# 	wr = csv.writer(csvFile, delimiter="\n")
	# 	wr.writerow(all_files)
	# csvFile.close()

	# #Reading file names to list written by Alex B.
	# with open("all_files.csv", 'r') as csvFile:
	# 	reader = csv.reader(csvFile, delimiter='\n')
	# 	all_files.append(reader)
	# csvFile.close()
	
	all_files = []
	with open('all_files.csv', newline='') as csvFile:
		for row in csv.reader(csvFile):
			all_files.append(row[0])

	all_files.sort()
	data_list = []
	sampling_rate_list = []

	#Writing feature sets to csv written by Alex B.

	# #write feature set from every file to csv
	# with open('features.csv', 'w') as csvFile:
	# 	writer = csv.writer(csvFile)
	# 	for file in all_files:
	# 		data, sr = librosa.load(file)
	# 		mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=features)
	# 		mfccsscaled = np.mean(mfccs.T,axis=0)
	# 		data_list.append(mfccsscaled)
	# 		writer.writerow(mfccsscaled)
	# csvFile.close()
	# print(data_list)

	#Optimal EPS value code written by Alex B.

	# #find the optimal eps value
	# neigh = NearestNeighbors(n_neighbors=4)
	# nbrs = neigh.fit(data_list)
	# distances, indices = nbrs.kneighbors(data_list)

	# distances = np.sort(distances, axis=0)
	# distances = distances[:,1]
	# plt.plot(distances)
	# plt.show()

	# #create list from csv
	# with open('features.csv', 'r') as f:
	# 	reader = csv.reader(f)
	# 	feature = list(reader)
	# 	data_list.append(feature)
	# 	#data_list.append(list(reader))
	# print(data_list)

	data_list = np.loadtxt('features_all.csv', delimiter=',')

	# data = file('features.csv').read()
	# table = [row.split(',') for row in data.split('\n')]
	# print(table)

	#turn lists into numpy arrays
	data_kmeans = data_list
	#data_list = np.array(data_list)
	sampling_rate_list = np.array(sampling_rate_list)

	#run DBSCAN on our data
	start_time = time.time()
	clustering = DBSCAN(eps=68, min_samples=2).fit(data_list)
	end_time = time.time()
	dbscan_elapsed_time = end_time - start_time
	print("Time DBSCAN: " + str(dbscan_elapsed_time))
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
	start_time = time.time()
	while (j < 5):
		k = (len(list_to_output))
		km = K_Means(k)
		kmeans_labels = km.fit(data_kmeans)
		j += 1
	end_time = time.time()
	our_kmeans_elapsed_time = end_time - start_time
	clusteringlist = kmeans_labels
	print("Time Kmeans: " + str(our_kmeans_elapsed_time))

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

	#scikit agglomerative output
	start_time = time.time()
	scikit_agg_labels = sci_kit_agg_clustering(data_list, len(list_to_output))
	end_time = time.time()
	scikit_agg_elapsed_time = end_time - start_time
	print("Time Agglomerative: " + str(scikit_agg_elapsed_time))

	#Setup for outputting to file
	zipped = zip(scikit_agg_labels, all_files)
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

	#scikit kmeans output
	start_time = time.time()
	scikit_kmeans_labels = sci_kit_KMeans(data_list, len(list_to_output))
	end_time = time.time()
	scikit_kmeans_elapsed_time = end_time - start_time
	print("Time Sci Kit Kmeans: " + str(scikit_kmeans_elapsed_time))

	#Setup for outputting to file
	clusteringlist = scikit_agg_labels

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

	count = 0
	resultmat = [[0 for x in range(len(data_list))] for y in range(len(data_list))]
	for i in clustering.labels_:

		for j in clustering.labels_:
			if clustering.labels_[i] == scikit_agg_labels[j]:
				count += 1
			if clustering.labels_[i] == scikit_kmeans_labels[j]:
				count += 1
			if clustering.labels_[i] == kmeans_labels[j]:
				count += 1
			resultmat[i][j]=count
			count = 0
	df = pd.DataFrame.from_records(resultmat)
	df.to_excel("output.xlsx")

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