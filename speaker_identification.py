from glob import glob
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import pickle
import nemo.collections.asr as nemo_asr
from statistics import mean
import csv

# from memory_profiler import profile

# instantiating the decorator
# @profile

'''
speaker identification for a audio sample on a trained model.
'''
def speaker_identification(input_file,Pkl_Filename,ECAPA_TDNN):

    # ecapa_model_start = time.time()
    # ECAPA_TDNN = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn')
    # ecapa_model_end = time.time()

    start_extract = time.time()
    embs = ECAPA_TDNN.get_embedding(input_file) # extracting features for input .wav file
    audio_np = embs.detach().cpu().numpy()
    end_extract = time.time()

    start_model = time.time()
    with open(Pkl_Filename, 'rb') as file:
        model = pickle.load(file) # loading trained model
    end_model = time.time()

    start_prediction = time.time()
    prediction = model.predict(audio_np) # predicting
    end_prediction = time.time()

    time_list = [end_extract-start_extract, end_model-start_model, end_prediction-start_prediction]
    return prediction, time_list


if __name__ == '__main__':
    ecapa_model_start = time.time()
    ECAPA_TDNN = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn') # loading ECAPA TDNN model
    ecapa_model_end = time.time()
    ecapa_time = ecapa_model_end - ecapa_model_start

    #----------------------------------------------------------------
    'Uncomment this part for speaker recognitation for one utterance.'
    # input_file = input("Path to input file (.wav) : ")
    # Pkl_Filename = input("model path : ")
    # pred, time_list = speaker_identification(input_file, Pkl_Filename, ECAPA_TDNN)
    #
    # print("ECAPA loading time : ", ecapa_time)
    # print("Feature Extraction time : ",time_list[0])
    # print("Model loading time : ",time_list[1])
    # print("Prediction time : ",time_list[2])
    # print("Total time : ",ecapa_time + sum(time_list) )
    # print("Prediction : ", pred)
    #----------------------------------------------------------------

    # main function to calculate average time taken for all parts given a model and number of digits in test files.
    #----------------------------------------------------------------
    data_path = input("Path to data folder : ")
    n = int(input("number of digits in test : "))
    model_path = input("model path : ")

    test_files = glob(data_path+"/*/*D/merged_audios/"+str(n)+"/*.wav" )
    extract_times = []
    model_times = []
    predict_times = []

    for input_file in test_files:
        pred,time_list = speaker_identification(input_file,Pkl_Filename,ECAPA_TDNN)
        extract_times.append(time_list[0])
        model_times.append(time_list[1])
        predict_times.append(time_list[2])

    print("Model : ", Pkl_Filename.split('/')[-1], " Average time : ",mean(time_taken) )
    print("ECAPA loading time : ", ecapa_time)
    print("Average Extracting time : ", mean(extract_times))
    print("Average model loading time : ", mean(model_times))
    print("Average prediction time : ", mean(predict_times))
    print("Average total time : ", ecapa_time + mean(extract_times) + mean(model_times) + mean(predict_times))
    #--------------------------------------------------------------
