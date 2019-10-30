import glob
import os
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
    for file in all_files:
        data_list.append(librosa.load(file))
        sampling_rate_list.append(librosa.load(file))

    #will extract 40 Mel-frequency cepstral coefficients
    #these can be used as features to represent each wav file
    #to learn more about MFCCS visit https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
    
    mfccs = librosa.feature.mfcc(y=data_list, sr=sampling_rate_list, n_mfcc=40)

    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    print(mfccsscaled)

if __name__ == "__main__":
    main()