import librosa
import os
import soundfile as sf
from pathlib import Path
import sys

Path('Audio_16k').mkdir(exist_ok=True)  #exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常
# IEMOCAP_DIR = Path(sys.argv[1])
print ("Downsampling IEMOCAP to 16k")
# for i in range(5):
#     sess = i + 1
#     current_dir = IEMOCAP_DIR / f"Session{sess}" / "sentences" / "wav"
#     for full_audio_name in current_dir.rglob('*.wav'):            #可以用pathlib模块中的Path().rglob来递归遍历文件

filename = "/home/lqb/project/SSL_Anti-spoofing-main/data/train/wav.scp"
with open(filename, 'r') as raw:
    for  full_audio_name in raw:
        audio_name = full_audio_name.split(" ")[0] + ".wav"
        path_audio_name = full_audio_name.split(" ")[1]
        print("audio_name=",audio_name)
        print("path_audio_name=",path_audio_name)
        path_audio_name = path_audio_name.replace('\n','')
        audio, sr = librosa.load(str(path_audio_name), sr=None)  # 其中的load函数就是用来读取音频的。当然，读取之后，转化为了numpy的格式储存，而不再是音频的格式了
        # audio_name = full_audio_name.name
        assert sr == 16000
        sf.write(os.path.join('Audio_16k/train', audio_name), audio, 16000)  #音频soundfile用来存储wav，flac，ogg等格式文件特别便捷

