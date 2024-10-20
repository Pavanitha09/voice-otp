from glob import glob
import numpy as np
import re
from sklearn.metrics import accuracy_score
import time
import pickle
import statistics

# test_model tests test_data on given model
# model_path : STRING - path to the saved model in .pkl format
# test_data : LIST - list of xvectors for testing
# test_labels : LIST - list of actual label for respective xvector in test_data
# returns time taken for model loading, average time taken to predict, accuracy, list of predicted labels

def test_model(model_path, test_data, test_labels):
    print("testing...")

    load_start = time.time()
    with open(model_path, 'rb') as file: # loading model
        model = pickle.load(file)
    load_end = time.time()
    load_time = load_end - load_start

    test_times = []
    prediction = []
    for test_file in test_data:
        test_start = time.time()
        prediction.extend(model.predict(test_file))
        test_end = time.time()
        test_times.append(test_end - test_start)

    avg_test_time = statistics.mean(test_times)
    accuracy = accuracy_score(test_labels, prediction)

    return load_time, avg_test_time, accuracy, prediction

if __name__ == '__main__':

    folder_path = "./data/" # path to folder containing speakers data
    folders = glob(folder_path+"/[0-9]*/")
    folders.sort()

    n = input("number of digits : ") # number of digits in utterances for training
    model_path = input("model path : ") # path to saved model

    test_data = []
    test_labels = []

    # getting testing data and respective labels (D) session for each speaker
    for speaker_path in folders:
        l = glob(speaker_path+"/*['D']/"+str(n)+'/*.npy')
        l.sort()
        label = re.split(r'/|\|//|\\', speaker_path)[-2]

        for vector_path in l:
            file = np.load(str(vector_path))
            test_data.append(file)
            test_labels.append(label)

    load_time, avg_test_time, accuracy, pred_labels = test_model(model_path, test_data, test_labels) # testing model

    print("number of digits : ",n)
    print("Model loading time : ", load_time)
    print("avg_test_time : ",avg_test_time)
    print("accuracy : ", accuracy)
