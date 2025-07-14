# Visual Speech Recognition: AI-Based Lip Reading with DNNs

This project builds a visual speech recognition (VSR) system that interprets speech using only lip movements, without audio. It uses Deep Neural Networks, combining 3D CNNs and Bidirectional GRUs, and achieves high accuracy in real-time lip-reading tasks.

## 🎯 Objectives

- Build a deep learning model that recognizes speech from visual lip movements.
- Achieve high classification accuracy with a self-collected dataset.
- Enable real-time, speaker-independent lip-reading.
- Ensure generalization to various speakers and environmental conditions.

## Training Data

You can view the data in the ```/collected_data/``` folder.

Since a suitable dataset was not available for this problem, I took the initiative to create my own dataset by collecting approximately 700 video clips of words being spoken. Each video clip was manually labeled with a word from a predefined set and in the end, I had around 3 gigabytes worth of total data.

How does my `/data_collection/collect.py` script work? 
- First detect if someone is talking (based upon if the distance between the upper lip and lower lip is greater than a threshold)
- If someone is talking:
  - Start "recording" or storing the current frames
  - The user can close their mouths for some short periods of time, but if they close their lips for too many frames then the script assumes they have finished talking.
  - Once the user finishes talking, we stop saving frames and then pad the current saved frames with "previous" and "after" frames
    - "previous" frames are frames stored (in a circular buffer) before we started saving frames
    - "after" frames are just continously added until we reach 22 frames (a predefined constant)
  - all 22 frames are stored in a numpy array and written to a text file (we also record the frames as images and turn those into a video file)
 

## Files/Folders

- `/data_collection/collect.py`: This script is used to collect the data for training the speech recognition model. It records audio clips of people speaking the different commands and saves them to a directory.

- `/training/3DCNN.ipynb`: This Jupyter Notebook contains the code used to train the speech recognition model. It uses a 3D convolutional neural network to extract features from the audio clips and then classifies them using a softmax layer.

- `/demo/predict_live.py`: This script can be used to test my trained speech recognition model in a live demo. It uses similar logic from the data collection script by collecting frames, feeding it into the model, and then displays the predicted command.

- `/model/`: contains various model weights for trained models.

- `/demo_examples/`: mp4 files containing recorded demos of my model predicting words that I spoke in real-time.


## Usage

To use my project, please read the following:

1. Run `collect.py` to record audio clips of yourself speaking a chosen word and save them to a directory.
    - First, enter the word you want to collect data for.
    - If this is your first time running the script, do not enter a lip distance (it will calibrate itself).
2. View `/training/3DCNN.ipynb` to see how I trained the speech recognition model on my collected data.
3. Use `predict_live.py` to test the trained model in a live demo. 

Note: The current model implemented in `predict_live.py` is a less accurate model. I am unable to upload the weights for the model showcased in `/training/3DCNN.ipynb` as the file is too large.

## Dependencies

The following Python packages required to run the scripts can be found in ```requirements.txt```. There might also be some issues relating to OS/OS versions. I am running this project on a macOS Catalina with a Intel Core i7 processor.

## 🛠️ Architecture Overview

Input Video → Preprocessing → 3D CNN → Bi-GRU → CTC Decoder → Text Output


## ⚠️ Note

This repository does **not** include the full dataset or trained model weights due to GitHub's size limitations.

If you're interested in accessing:

- The full `collected_data/` (~3 GB of custom video clips)
- The trained model files in `model/`

Please reach out to me at 📧 **shreeyamane173@gmail.com** and I’ll be happy to share them with you.

And To use the live prediction script `/demo/predict_live.py`, please note that it may not work on all systems. You may need to modify the script to work with your specific setup, including having the same operating system and webcam as the one used during training.


