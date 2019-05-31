import numpy as np
import scipy
import matplotlib.pyplot as plt

import sklearn.cluster
from scipy.signal import savgol_filter
import librosa
import librosa.display
from scipy import signal

BINS_PER_OCTAVE = 12 * 3
N_OCTAVES = 7
    
plot =True
def getLogPowCQT(y,sr):
    C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                                            bins_per_octave=BINS_PER_OCTAVE,
                                            n_bins=N_OCTAVES * BINS_PER_OCTAVE),
                                ref=np.max)
    return C

def smoothify(thisarray):
    """
    returns moving average of input using:
    out(n) = .7*in(n) + 0.15*( in(n-1) + in(n+1) )
    """

    # make sure we got a numpy array, else make it one
    if type(thisarray) == type([]): thisarray = np.array(thisarray)

    # do the moving average by adding three slices of the original array
    # returns a numpy array,
    # could be modified to return whatever type we put in...
    return 0.7 * thisarray[1:-1] + 0.15 * ( thisarray[2:] + thisarray[:-2] )


path = 'C:\\Users\\peerfunk\\Documents\\bac\\Analysis\\WavToCSV\\visualStudio\\wavGetterDotNET\\wavGetterDotNET\\bin\\Release\\'
y, sr = librosa.load(path + 'out.wav')

print(len(y))
C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
                n_bins=60 * 2, bins_per_octave=12 * 4))

tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
Csync = librosa.util.sync(C, beats, aggregate=np.median)



if True:
    plt.figure(figsize=(12, 4))
    plt.plot(y)
    librosa.display.specshow(C, y_axis='cqt_hz', sr=sr,
                             bins_per_octave=BINS_PER_OCTAVE,
                             x_axis='time')
    plt.tight_layout()
plt.show()


mean =np.mean(Csync)*1.4 #over 90%
filtered = np.where(Csync > mean,Csync, -80)
filteredStripes=filtered[list(range(10, 20))+list(range(50, 80)),:]
columns = filteredStripes.sum(axis=0)
print(len(columns))
if len(columns)> 10:
    isStomp=savgol_filter(columns, 11, 1)
else:
    isStomp=[0]
isStompMedian=np.mean(isStomp)
isStomp+=abs(isStompMedian)
print(abs(isStompMedian))
selected = np.where(isStomp>(np.max(isStomp)*0.5),isStomp,0)
np.savetxt("SelectedOut.csv", isStomp, delimiter=";")
if True:
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(filtered, bins_per_octave=12*4,
                             y_axis='cqt_hz', x_axis='time')
    plt.show()
    plt.figure(figsize=(12, 4))
    plt.plot([0]*len(isStomp))
    plt.plot(selected)
    plt.plot(isStomp)
    plt.show()





