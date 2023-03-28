from pydub import AudioSegment
from pydub.silence import split_on_silence
from glob import glob
import os
from itertools import product
import random


'''
digit_generate function is to generate all possible n digit number combinations.

INPUT:
# speaker_path : STRING
path to directory which contains folders of all sessions (A,B,C,D) and in each session folder has utterances of 0 to 9 digits as .wav files
(Example : speaker_path/A/0.wav)

# number_of_digits : STRING
number of digits that should be in each combination.

OUTPUT: creates a directory "merged_audios/{number_of_digits}" in each session folder which contains all possible combinations.

'''

def digit_generator(speaker_path, number_of_digits):

    print(speaker_path)
    sessions = glob(speaker_path+"/0*/")  # getting a list with paths to all session folders inside the speaker folder
    sessions.sort()

    for session in sessions: # for each session

        try:
            os.makedirs(session + 'merged_audios') # creating merged_audios directory if it does not exists
        except OSError:
            pass

        files = glob(session+"/*[0-9].wav") # getting a list with path to all audio files.
        files.sort()

        try:
            os.makedirs( session + "merged_audios/" + str(number_of_digits)) # creating {number_of_digits} directory inside merged_audios directory
        except OSError:
            pass

        combs = product(files, repeat=number_of_digits) # all possible n digits combinations possibl
        for comb in combs:
            file_savepath = session + "merged_audios/" + str(number_of_digits) + "/"
            segments = []
            for file in comb:
                file_savepath += str(file[-5])
                audio = AudioSegment.from_file(file)
                silence = AudioSegment.silent(duration=200)  # 1 second of silence
                segments.append(audio + silence)

            file_savepath += '.wav'     # Concatenate the audio segments
            output = segments[0]
            for segment in segments[1:]:
                output = output + segment
            output.export(file_savepath, format="wav")     # Export the concatenated audio as a new file


'''
random_digit_generate function is to generate given number of random possible n digit number combinations.

INPUT:
# speaker_path : STR
path to directory which contains folders of all sessions (A,B,C,D) and in each session folder has utterances of 0 to 9 digits as .wav files
(Example : speaker_path/A/0.wav)

# number_of_digits : INT
number of digits that should be in each combination.

# number_of_combinations : INT
number of combinations required

#seed_value: INT
seed value

OUTPUT: creates a directory "merged_audios/{number_of_digits}" in each session folder which contains random combinations.
'''

def random_digit_generator(speaker_path, number_of_digits, number_of_combinations, seed_value):
    print(speaker_path)
    sessions = glob(speaker_path+"/0*/")  # getting a list with paths to all session folders inside the speaker folder
    sessions.sort()
    i = 0
    for session in sessions: # for each session

        try:
            os.makedirs(session + 'merged_audios')         # creating merged_audios directory if it does not exists
        except OSError:
            pass

        files = glob(session+"/*[0-9].wav")     # getting a list with path to all audio files.
        files.sort()

        try:
            os.makedirs( session + "merged_audios/" + str(number_of_digits)) # creating {number_of_digits} directory inside merged_audios directory
        except OSError:
            pass

        combs = product(files, repeat=number_of_digits) # all possible n digits combinations possible
        random.seed(seed_value+i)
        combs = random.sample(combs, number_of_combinations) # from all possible combinations picking n unquie values

        for comb in random_list:
            file_savepath = session + "merged_audios/" + str(number_of_digits) + "/"
            segments = []
            for file in comb:
                file_savepath += str(file[-5])
                audio = AudioSegment.from_file(file)
                silence = AudioSegment.silent(duration=200)  # 1 second of silence
                segments.append(audio + silence)

            file_savepath += '.wav' # Concatenate the audio segments
            output = segments[0]
            for segment in segments[1:]:
                output = output + segment
            output.export(file_savepath, format="wav") # Export the concatenated audio as a new file

        i+=1


if __name__ == '__main__':

# Main function generates for all speakers in the data folder for to generate all possible for number_of_digits taken as input
#---------------------------------------------------------------------
number_of_digits = int(input("number of digits : "))
speakers = glob("data/*/")
speakers.sort()
for speaker_path in speakers:
    print(speaker_path)
    digit_generator(speaker_path,number_of_digits)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
"Uncomment this code for random_digit_generator for all speakers in data folder"
# number_of_digits = int(input("number of digits : "))
# number_of_combinations = int(input("number of combintions : "))
# seed_value = int(input("seed : "))
# speakers = glob("data/*/")
# speakers.sort()
# for speaker_path in speakers:
#     print(speaker_path)
#     random_digit_generator(speaker_path,number_of_digits,number_of_combinations,seed_value)
#---------------------------------------------------------------------

#----------------------------------------------------------------------
"uncomment this part for generating for only desired speakers"
# t = int(input("Number of speakers : "))
#
# for _ in range(t):
#     speaker_path = input("speaker_path : ")
#     number_of_digits = int(input("number of digits : "))
#
#     digit_generate(speaker_path,number_of_digits)
#----------------------------------------------------------------------
