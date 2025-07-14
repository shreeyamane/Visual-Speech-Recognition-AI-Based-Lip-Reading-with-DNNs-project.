# Visual Speech Recognition: AI-Based Lip Reading with DNNs

This project builds a visual speech recognition (VSR) system that interprets speech using only lip movements, without audio. It uses Deep Neural Networks, combining 3D CNNs and Bidirectional GRUs, and achieves high accuracy in real-time lip-reading tasks.

🎯 Objectives

- Build a deep learning model that recognizes speech from visual lip movements.
- Achieve high classification accuracy with a self-collected dataset.
- Enable real-time, speaker-independent lip-reading.
- Ensure generalization to various speakers and environmental conditions.


🛠️ Architecture Overview

Input Video → Preprocessing → 3D CNN → Bi-GRU → CTC Decoder → Text Output
