import numpy as np
import librosa
# import soundfile as sf
import glob
import re
def loader(x,path=None,suffix=None):
        x=librosa.stft(x, n_fft = n_fft, hop_length = hop_length)
        x=np.abs(x).transpose()    
        end=x.shape[0]-x.shape[0]%10
        x=x[:end]
        L=[]
        x=np.reshape(x,(-1,10,513))
        dir=path.replace(r"C:\Users\Axiomatize\Desktop\MIR-1K\mir-1k\Wavfile",rf"E:\Mir\{suffix}").replace(".wav",f" {suffix}.npy")
        
        dir=re.sub(special, "", dir)
        np.save(dir,x)

base=r"C:\Users\Axiomatize\Desktop\MIR-1K\mir-1k\Wavfile\*.wav"
files =glob.glob(base)

special =r"[!|@|#|$|%|^|&|*|(|)|_|'|:|;|?]"

sr=16000
n_fft = 1024
hop_length = n_fft // 4
sample_frames = 10
k=1
for path in files:
    print(k)
    k+=1    
    mix,_=librosa.load(path, sr = sr, mono = False)
    mel = mix[0, :]
    voc = mix[1, :]
    mix=librosa.to_mono(mix) * 2
    
    
    loader(mix,path,'mix')
    loader(voc,path,'voc')
    loader(mel,path,'mel')


# sf.write(r"C:\Users\Axiomatize\Desktop\voc.wav",voc,samplerate=sr)
