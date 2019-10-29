import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa.feature

data, sampling_rate = librosa.load('./data/203424.wav')

#will extract 40 Mel-frequency cepstral coefficients
#these can be used as features to represent each wav file
#to learn more about MFCCS visit https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40)
mfccsscaled = np.mean(mfccs.T,axis=0)
print(mfccsscaled)
