import librosa
import os
import sys
import pickle as p
import numpy as np
BINS_PER_OCTAVE = 12 * 3
N_OCTAVES = 7
    
def getAllFiles(path,FileType):
    ret=[]
    for file in os.listdir(path):
        if file.endswith(FileType):
            if path is None:
                 path=""
            curfile = os.path.join(path, file)
            ret.append(curfile)
    return ret
def WavToCQT(filePathName):
    y, sr = librosa.load(filePathName[1])
    #C = librosa.amplitude_to_db(librosa.cqt(y=y,sr=sr,bins_per_octave=BINS_PER_OCTAVE,n_bins=N_OCTAVES * BINS_PER_OCTAVE),ref=np.max)
    return np.abs(librosa.stft(y))
def CQTToNP(data,fileName):
    p.dump(data,open(fileName, "wb"))
def WavToNP(filePathName):
    p.dump(WavToCQT(filePathName), open(filePathName[0]+".npy","wb"))
def AllWavtoNP(path=None):
    for file in getAllFiles(path, Filetype):
        WavToNP(file)
def WavToCQTList(path=None, Filetype=".wav"):
    ret=[]
    for file in getAllFiles(path, Filetype):
        ret.append(WavToCQT(file))
    return ret
def NPtoCQTList(path):
    ret=[]
    for file in getAllFiles(path, ".npy"):
        ret.append(p.load(open(file[1], 'rb')))
    return ret
def getMinLength(files):
    minLength = 0
    for file in files:
        print(file)
        y, sr = librosa.load(file)
        if minLength > len(y) or minLength ==0:
            minLength = len(y)
        print(minLength)
    return minLength

        
#Csync = p.load(open("csync.npy", 'rb'))
