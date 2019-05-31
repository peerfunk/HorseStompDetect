import pyaudio
import wave
import queue
import threading
import time
import numpy as np
import librosa
import pickle as p
import matplotlib.pyplot as plt
import logging
import librosa.display
FORMAT = pyaudio.paInt16
CHANNELS =2
RATE = 44100
RECORD_SECONDS = 10
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()

AudioImputQ=queue.Queue()
FFTQ= queue.Queue()
PlotQ = queue.Queue()

def getPattern(path):
    return p.load(open(path, "rb"))

def getStartEnd(data,start , end):
    return np.array(data[:,list(range(start,end))])
def SlidingWindow(Pattern, SearchAble,WindowWidth,AlertFunction):
    #PlotQ.put(SearchAble)
    start_time = time.time()
    ret=[0]
    ret2=[]
    for sliding in range(1,len(SearchAble[1])-WindowWidth):
        #start_time2 = time.time()
        Frame2=getStartEnd(SearchAble,sliding,sliding+WindowWidth)
        #elapsed_time = time.time() - start_time2
        #print("slidingWindow - getStartEnd:" + str(elapsed_time))
        #start_time2 = time.time()
        diff = Pattern - Frame2
        count=(diff<0).sum()
        cellCount = len(diff) * len(diff[0])
        perc = count/cellCount
        #elapsed_time = time.time() - start_time2
        #print("slidingWindow - calc:" + str(elapsed_time))
        ret.append(perc)
        if perc > 0.90:
            ret2.append(1)
        else:
            ret2.append(0)
    retrn=filter(ret2,RECORD_SECONDS)
    if len(retrn)>=2:
        AlertFunction()
    elapsed_time = time.time() - start_time
    PlotQ.put(ret)
    print(retrn)
    print("slidingWindow, all:" + str(RECORD_SECONDS) + ";" + str(elapsed_time) + ";"  + time.strftime("%H:%M:%S", time.localtime(start_time)) + ";"  + time.strftime("%H:%M:%S", time.localtime(time.time())))
def filter(result,duration):
    counter=0
    ret=[]
    ret1=[]
    last=0
    durPerCell=duration/len(result)
    for x,i in enumerate(result):
        if i==1:
            if last == 0:
                ret.append(x)
            counter+=1
        else:
            if last==1:
                ret.append(x)
                counter =0
        last = i
    for j in range(0,len(ret)-2):
        space = ((ret[j+1]-ret[j])*durPerCell)
        if space < 4 and space > 0.6:
            #print(str(ret[j+1]-ret[j]) +"," + str(ret[j]))
            ret1.append(ret[j])
    return ret1

def AudioImportWorker():
    frame = np.array([])
    bar = int(RATE * RECORD_SECONDS)
    n = 0
    while True:
        if not AudioImputQ.empty():
            start_time = time.time()
            data = AudioImputQ.get()
            #print("Get item from AudioImputQ n=" + str(AudioImputQ.qsize()) )
            frame = np.append(frame,data)
            if len(frame) > bar:
                frame = librosa.util.normalize(frame)
                start_time = time.time()
                elapsed_time = time.time() - start_time
                #print("AudioImportWorker - contact:" + str(elapsed_time))
                start_time = time.time()
                fft = np.abs(librosa.stft(frame))**2
                #print("Added item to FFTQ n=" + str(FFTQ.qsize()) )
                FFTQ.put(fft)
                frame = np.empty(1)
                elapsed_time = time.time() - start_time
                #print("AudioImportWorker  - fft:" + str(elapsed_time))
                n+=1
def AnalyzeWorker():
    Pattern = getPattern("Pattern.npy")
    while True:
        if not FFTQ.empty():
            start_time = time.time()
            #print("Get item from FFTQ n=" + str(FFTQ.qsize()))
            fftqData = FFTQ.get()
            x = threading.Thread(target=SlidingWindow, args=(Pattern,fftqData,len(Pattern[0]),Alert))
            x.start()
            #PlotQ.put(result2)
            elapsed_time = time.time() - start_time
            #print("AnalyzeWorker:" + str(elapsed_time))


def callback(in_data, frame_count, time_info, flag):
    data = np.fromstring(in_data, dtype=np.float32)
    AudioImputQ.put(librosa.util.normalize(data))
    #print("Added item to AudioImputQ n=" + str(AudioImputQ.qsize()) ) 
    return None, pyaudio.paContinue
def Alert():
    print("!!!!ALERT!!!!")
def main():           
    t1 = threading.Thread(target=AudioImportWorker)
    t1.start()
    t2 = threading.Thread(target=AnalyzeWorker)
    t2.start()
    t3 = threading.Thread(target=WatchDog)
    t3.start()
    stream = audio.open(format=pyaudio.paFloat32,
                     channels=CHANNELS,
                     rate=RATE,
                     output=False,
                     input=True,
                     frames_per_buffer=CHUNK,
                     stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
        if not PlotQ.empty():
            plt.figure()
            plt.plot(PlotQ.get())
            #librosa.display.specshow(librosa.amplitude_to_db(PlotQ.get(),ref=np.max),y_axis='log', x_axis='time')
            plt.show()
    stream.close()
    audio.terminate()
def WatchDog():
    while True:
        time.sleep(0.01)
        #print("AudioImputQ:" + str(AudioImputQ.qsize()) + "FFTQ:"  + str(FFTQ.qsize()) +"PlotQ:" + str(PlotQ.qsize()))
main()
