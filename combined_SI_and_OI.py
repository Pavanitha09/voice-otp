from glob import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import pickle
import nemo.collections.asr as nemo_asr
from statistics import mean
import sys

ECAPA_TDNN = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn')
VAKYANSH = nemo_asr.models.ASRModel.restore_from("Conformer-CTC-BPE-Large.nemo")

# Path to generated trained model
with open(sys.argv[1], 'rb') as file:
    model = pickle.load(file)

text_to_digit = {}
text = ["zero","one","two","three","four","five","six","seven","eight","nine"]
for i in range(10):
    string = text[i]
    text_to_digit[i] = [string[start:start+length] for length in range(2, len(string) + 1) for start in range(len(string) - length + 1)]


def speaker_identification(input_file):
    embs = ECAPA_TDNN.get_embedding(input_file)
    audio_np = embs.detach().cpu().numpy()
    label = input_file.split('/')[-5]
    prediction = model.predict(audio_np)

    return label == prediction


def digit_recognization(input_file):
    print(input_file)
    transcription = VAKYANSH.transcribe([input_file])
    label = input_file.split('/')[-1].split('.')[0]
    pred =  ""
    for word in transcription[0].split():
        for digit in text_to_digit:
            if word in text_to_digit[digit]:
                pred+=str(digit)
                break

    return  label == pred


n = int(input("Enter the number of digits in test data"))
# Path to audio files
test_files = glob("path_to_audio_files" + str(n) + "/*.wav")
test_files.sort()
predictions = []

for input_file in test_files:
    s = speaker_identification(input_file)
    d = digit_recognization(input_file)
    predictions.append(s and d)

print(predictions.count(True)/len(predictions))