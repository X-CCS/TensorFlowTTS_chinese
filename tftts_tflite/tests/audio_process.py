# from ffmpeg import audio
# audio.a_speed("output_seq_tflite_after_24.wav", "0.5", "output_seq_tflite_after_24_new.wav")
# print("done")

# sox output_seq_tflite_after_24.wav -r 16000 output_seq_tflite_after_24_new_8.wav
# sox output_seq_tflite_after_24_new_3.wav -b 16 output_seq_tflite_after_24_new_4.wav

import librosa
# to install librosa package
# > conda install -c conda-forge librosa 

filename_22 = 'output_seq_tflite_after_22.wav'
filename_24 = 'output_seq_tflite_after_24.wav'
newFilename = 'output_seq_tflite_after_22_new.wav'

y_1, sr_1 = librosa.load(filename_22) # 22050
print("y_1:",y_1)
print("sr_1:",sr_1) # 22050
# y_2, sr_2 = librosa.load(filename_24) # 2400
# print("y_2:",y_2)
# print("sr_2:",sr_2) # 22050
# 不要重采样，重新采样会补点进去，还是原先的速度，直接强制调过，则可以降低
# y_12k = librosa.resample(y,sr,12000)
# 采样率 6000 需要更慢的语速择要更加低的采样率
librosa.output.write_wav(newFilename,y_1,6000) # 
print("done")