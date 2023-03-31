from glob import glob
import numpy as np
import math
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import pickle
import re


# train_model function trains a LogisticRegression model with given train data and train labels, saves model as .pkl file with given model_name
# train_data : List, test_labels : List, model_name : String

def train_model(train_data, train_labels, model_name):
    print("model training...")
    logreg = LogisticRegression(solver='lbfgs', max_iter=1000) # Create a logistic regression model

    train_start = time.time()
    logreg.fit(train_data, train_labels) # training model
    train_end = time.time()
    train_time = train_end-train_start

    with open(model_name, 'wb') as file:
        pickle.dump(logreg, file)

    return logreg, train_time


if __name__ == '__main__':
    folder_path = "./data_vectors/" # path to data folder
    folders = glob(folder_path+"/[0-9]*/") # returns list of path to all speaker folders ii data folder
    folders.sort()

    train_data = []
    train_labels = []

    n = input("number of digits in utterance for training : ")

    # getting training data and respective labels (A,B,C) sessions for each speaker
    for speaker_path in folders:
        l = glob(speaker_path+"/*['A','B','C']/"+str(n)+'/*.npy')
        l.sort()
        label = re.split(r'/|\|//|\\', speaker_path)[-2]
        print(label)
        for vector_path in l:
            file = np.load(str(vector_path))
            train_data.append(file[0])
            train_labels.append(label)

    try:
        os.makedirs("models")
    except:
        pass
    model_name = "./models/logreg"+n+".pkl"
    model, train_time = train_model(train_data, train_labels, model_name)

    print("Model training time :", train_time)
