import glob
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import librosa.display
import librosa.feature

def main():
    #get rid of scientific notation in numpy arrays
    np.set_printoptions(suppress=True)
    features = 8

    all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
    # print(all_files)
    all_files.sort()
    # print(len(all_files))
    data_list = []
    sampling_rate_list = []
    # for file in all_files:
    #     data_list.append(librosa.load(file))
    #     sampling_rate_list.append(librosa.load(file))

    #write feature set from every file to csv
    with open('features.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        for file in all_files:
            data, sr = librosa.load(file)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=features)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            data_list.append(mfccsscaled)
            # for x in mfccsscaled:
            #     x = round(x, 3)
            #     data_list.append(x)
            # print("scaled ", mfccsscaled)
            #mfccsscaled = round(mfccsscaled, 3)
            #data_list.append(mfccsscaled)
            writer.writerow(mfccsscaled)
    csvFile.close()

    # print("Pre numpy array data list: ", data_list)

    #turn lists into numpy arrays
    data_list = np.array(data_list)
    # for item in data_list:
    #     npdata = npdata.append(item, npdata[item])
    sampling_rate_list = np.array(sampling_rate_list)

    # print("Data list: ", data_list)
    # print("Sampling list: ", sampling_rate_list)

    testdata = np.array([[1, 2], [2, 2], [3, 5], [87, 99]])
    clustering = DBSCAN(eps=75, min_samples=2).fit(data_list)
    # print("Clustering: ", clustering.labels_)
    clustercount = np.max(clustering.labels_) + 1
    # print("cluster count is: ", clustercount)
    clusteringlist = list(clustering.labels_)
    # print(clusteringlist)
    clusters = np.empty([features, clustercount])
    #for i in range(clustercount):
    zipped = zip(clusteringlist, all_files)
    zipped = set(zipped)

    num_clusters = [[] for i in range(1, clustercount+1)]
    print(num_clusters)
    i = 0
    numiterations = 0
    for k,v in zipped:
        for i in range(len(zipped)):
            if k == i:
                num_clusters[i].append(v)
        i += 1
    print(num_clusters)
        
    #print("Clustering: ", clustering)

if __name__ == "__main__":
    main()