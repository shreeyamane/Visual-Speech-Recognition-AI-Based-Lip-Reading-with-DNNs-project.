import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
from collections import deque
from constants import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../model/face_weights.dat")
cap = cv2.VideoCapture(0)

# Data structures
all_words = []
curr_word_frames = []
labels = []
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)
not_talking_counter = 0
lip_distances = []

# Prompt user
words = ["here", "is", "a", "demo", "can", "you", "read", "my", "lips", "cat", "dog", "hello", "bye"]
label = input("What word you like to collect data for? The options are \n" + ", ".join(words) + ": ")
custom_distance = input("If you want, enter a custom lip distance threshold or -1: ")
clean_output_dir = input("To clean output directory of the current word, type 'yes': ")

# Clean output directory
if clean_output_dir.lower() == "yes":
    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    outputs_dir = os.path.join(root_dir, "outputs")
    for folder_name in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder_name)
        if os.path.isdir(folder_path) and label in folder_path:
            print(f"Removing folder {folder_name}...")
            os.system(f"rm -rf \"{folder_path}\"")

# Lip distance threshold logic
determining_lip_distance = 50
LIP_DISTANCE_THRESHOLD = None
if custom_distance != "-1" and custom_distance.isdigit() and int(custom_distance) > 0:
    LIP_DISTANCE_THRESHOLD = int(custom_distance)
    determining_lip_distance = 0
    print("USING CUSTOM DISTANCE")

# Main loop
data_count = 1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        if determining_lip_distance == 0 and LIP_DISTANCE_THRESHOLD is not None:
            # Padding
            width_diff = LIP_WIDTH - (lip_right - lip_left)
            height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
            pad_left = min(width_diff // 2, lip_left)
            pad_right = min(width_diff - pad_left, frame.shape[1] - lip_right)
            pad_top = min(height_diff // 2, lip_top)
            pad_bottom = min(height_diff - pad_top, frame.shape[0] - lip_bottom)

            lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
            lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

            # Enhancement pipeline
            lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lip_frame_lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
            l_eq = clahe.apply(l)
            lip_frame_eq = cv2.merge((l_eq, a, b))
            lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
            lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
            lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
            lip_frame = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)

            # Talking / Not Talking logic
            ORANGE = (0, 180, 255)
            BLUE = (255, 0, 0)
            RED = (0, 0, 255)

            if lip_distance > LIP_DISTANCE_THRESHOLD:
                cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                curr_word_frames.append(lip_frame.tolist())
                not_talking_counter = 0
                cv2.putText(frame, "RECORDING WORD RIGHT NOW", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
            else:
                cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                not_talking_counter += 1

                if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES:
                    curr_word_frames = list(past_word_frames) + curr_word_frames
                    all_words.append(curr_word_frames)
                    labels.append(label)
                    print(f"adding {label.upper()} shape", lip_frame.shape, "count is", data_count, "frames is", len(curr_word_frames))
                    data_count += 1
                    curr_word_frames = []
                    not_talking_counter = 0

                elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                    curr_word_frames.append(lip_frame.tolist())
                    not_talking_counter = 0

                elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                    curr_word_frames = []

                past_word_frames.append(lip_frame.tolist())

        else:
            # Calibrate lip distance
            cv2.putText(frame, "KEEP MOUTH CLOSED, CALIBRATING DISTANCE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            determining_lip_distance -= 1
            distance = landmarks.part(58).y - landmarks.part(50).y
            cv2.putText(frame, "Current distance: " + str(distance + 2), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            lip_distances.append(distance)
            if determining_lip_distance == 0:
                LIP_DISTANCE_THRESHOLD = sum(lip_distances) / len(lip_distances) + 2

    cv2.putText(frame, f"COLLECTED WORDS: {len(all_words)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'ESC' to exit", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Mouth", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        print("ESC pressed. Exiting and saving data...")
        break

# Save all words
def saveAllWords(all_words):
    print("saving words into dir!")
    output_dir = "../collected_data"
    next_dir_number = 1

    for i, word_frames in enumerate(all_words):
        label = labels[i]
        word_folder = os.path.join(output_dir, f"{label}_{next_dir_number}")
        while os.path.exists(word_folder):
            next_dir_number += 1
            word_folder = os.path.join(output_dir, f"{label}_{next_dir_number}")
        os.makedirs(word_folder)

        with open(os.path.join(word_folder, "data.txt"), "w") as f:
            f.write(json.dumps(word_frames))

        images = []
        for j, img_data in enumerate(word_frames):
            img = Image.new('RGB', (len(img_data[0]), len(img_data)))
            pixels = img.load()
            for y in range(len(img_data)):
                for x in range(len(img_data[y])):
                    pixels[x, y] = tuple(img_data[y][x])
            img_path = os.path.join(word_folder, f"{j}.png")
            img.save(img_path)
            images.append(imageio.imread(img_path))

        video_path = os.path.join(word_folder, "video.mp4")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10
        imageio.mimsave(video_path, images, fps=fps)
        next_dir_number += 1

        # Preview video
        video = cv2.VideoCapture(video_path)
        if video.isOpened():
            print(f"Previewing video for '{label}' — press ESC to close.")
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                cv2.imshow("Preview - " + label, frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
            video.release()
            cv2.destroyWindow("Preview - " + label)

# Final cleanup
cap.release()
cv2.destroyAllWindows()
saveAllWords(all_words)
