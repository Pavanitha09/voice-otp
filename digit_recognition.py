from sklearn.metrics import accuracy_score
import nemo.collections.asr as nemo_asr
from glob import glob
import sys
import time
import os
from numpy import asarray
from numpy import save

text_to_digit = {}
text = ["zero","one","two","three","four","five","six","seven","eight","nine"]
for i in range(len(text)):
    string = text[i]
    text_to_digit[i] = [string[start:start+length] for length in range(2, len(string) + 1)
                           for start in range(len(string) - length + 1)]

time_start = time.time()
model = nemo_asr.models.ASRModel.restore_from("Conformer-CTC-BPE-Large.nemo")
time_end = time.time()
n = int(input("Enter the number of digits in the test sample"))
files = glob("/path_to_audio_files/"+ str(n) +"/*.wav")

files.sort()
# print(files[10])
labels = []
predictions = []
# Transcribe an audio file
path = os.getcwd()
try:
    os.makedirs(path + '/labels')
except:
    print("Already exists")
count = 0
times = []
for audio_file in files:
    print(audio_file)
    transcription = model.transcribe([audio_file])

    label = audio_file.split('/')[-1].split('.')[0]
    labels.append(label)
    pred = ""
    for word in transcription[0].split():
        for digit in text_to_digit:
            if word in text_to_digit[digit]:
                pred+=str(digit)
                break

    predictions.append(pred)
digit_labels = []
for i in range(len(labels)):
    if(predictions[i] == labels[i]):
        digit_labels.append(True)
    else:
        digit_labels.append(False)
digit_rg_labels = asarray([digit_labels])
file_name = path + "/labels/pred_labels_" + str(n) + ".npy"
save(file_name, digit_rg_labels)
