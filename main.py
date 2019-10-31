import glob
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.feature

def main():
    all_files = glob.glob('assignment5/data/*.wav') #assignment5/data
    all_files.sort()
    # print(len(all_files))
    data_list = []
    sampling_rate_list = []
    # for file in all_files:
    #     data_list.append(librosa.load(file))
    #     sampling_rate_list.append(librosa.load(file))

    #turn lists into numpy arrays
    data_list = np.asarray(data_list)
    sampling_rate_list = np.asarray(sampling_rate_list)

    #write feature set from every file to csv
    with open('features.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        for file in all_files:
            data, sr = librosa.load(file)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=10)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            writer.writerow(mfccsscaled)
    csvFile.close()

if __name__ == "__main__":
    main()