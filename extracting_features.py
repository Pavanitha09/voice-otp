from glob import glob
import os
import re
import nemo.collections.asr as nemo_asr
import numpy as np
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='ecapa_tdnn')

'''
The extracting features function is to generate a .npy file for a given audio file

INPUT:
1) speaker_audio file
FORMAT : STRING

2) Path of the folder where the numpy files are gonna be stored

3) number of digits in the audio file
FORMAT : STRING

OUTPUT : creates a directory new directory for storing vectors where the directory structure is of the form : "{folder_name}/{speaker_number}/{session}/{number_of_digits}" 

(Example: data_vectors/001/A/2/50.npy)

'''

def extracting_features(speaker_path , vector_path, number_of_digits):

    # Creating a folder for each speaker in the vectors folder
    
    vectors_speaker_path = vector_path + '/' + re.split(r'/|\|//|\\', speaker_path)[-1] + '/'
    print(vectors_speaker_path)
    try:
        os.makedirs(vectors_speaker_path)
    except:
        pass
    speaker_sessions = glob(speaker_path+"/*/")
    speaker_sessions.sort()
    for session in speaker_sessions:

        # Creating a folder for each session in vectors folder

        vector_session = vectors_speaker_path + re.split(r'/|\|//|\\', session)[-2] + '/'
        try:
            os.makedirs(vector_session)
        except:
            pass

        # Creating the digits folder in each session
    
        digits_folder = vector_session + str(number_of_digits) + '/'
        try:
            os.makedirs(digits_folder)
        except:
            pass
        
        audios_files_combinations = glob(session + 'merged_audios/' + str(number_of_digits) + '/*.wav')
        audios_files_combinations.sort()
        for combination in audios_files_combinations:
            vec_name =  digits_folder + re.split(r'/|\|//|\\', combination)[-1][:-4] + '.npy'
            print(vec_name)
            embs = speaker_model.get_embedding(combination)
            xvector = embs.detach().cpu().numpy()
            np.save(vec_name, xvector)




if __name__ == '__main__':

# Main function generates numpy files for all the audio files in the data folder
#---------------------------------------------------------------------
    number_of_digits = input("number of digits : ")
    speakers = glob("./data/*")
    speakers.sort()

    # To create a folder just to store numpy files if it doesnot exist

    vector_path = os.getcwd() + '/data_vectors'
    try:
        os.makedirs(vector_path)      
    except OSError:
        pass
    print(vector_path)

    for speaker_path in speakers:
        print(speaker_path)
        extracting_features(speaker_path,vector_path, number_of_digits)
