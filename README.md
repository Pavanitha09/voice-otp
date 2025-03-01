# Voice OTP Authentication System

## Setup

1. **Create a Python environment** and install the required packages using:
   ```bash
   pip install -r requirements.txt

## Components

**1. OTP Generation**
    - digit_generator.py: Generates OTPs from audio files.

**2. Speaker Identification**
* extracting_features.py: Extracts x-vectors (saved as .npy files) from audio files using the ECAPA-TDNN model.
* logreg_model_train.py: Trains a logistic regression model using the x-vectors and saves the trained model as a .pkl file.
* model_test.py: Tests the speaker identification model on the test dataset.
* speaker_identification.py: Related to the time taken for speaker_identification.
    
**3. Digit Recognition**
* digit_recognition.py: Uses a Conformer model for digit recognition.\
* Link: https://github.com/Open-Speech-EkStep/vakyansh-models#finetuned-asr-models \
**Note**: For digit recognition part, download the vakyansh finetuned Conformer-based Indian English model from the above link.

**4. Voice OTP Authentication**
* combined_SI_and_OI.py: Contains the full workflow for voice OTP authentication, combining both speaker identification and digit recognition.

## Finetuning

   **Speaker Identification Finetuning**:
   * The Finetuning folder contains files related to ECAPA-TDNN model finetuning.
   * To use a finetuned ECAPA-TDNN model and do speaker identification, follow the same steps as above, replacing the model path with the finetuned version.\
Link: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ecapa_tdnn \
Refer to this link for more details

