# Lip Reading with Deep Learning

This project implements a lip reading application using deep learning techniques, inspired by the LipNet research paper.

## Overview

The application interprets lip movements from videos and converts them into text. It comprises three main modules:

1. **modelutil.py**: Contains the lip reading model architecture and pre-trained weights loading.

2. **utils.py**: Provides data preprocessing and loading utilities for videos and alignment data.

3. **streamlitapp.py**: Implements a Streamlit-based web application for interacting with the lip reading model.

## Instructions

1. **Setup**
   - Ensure necessary dependencies (TensorFlow, Streamlit, OpenCV, etc.) are installed.
   - Place video files in the specified directory (`../data/s1/`).

2. **Usage**
   - Run `streamlitapp.py` to start the Streamlit web application.
   - Select a video from the provided options for lip reading analysis.

3. **Interaction**
   - View the selected video in the application.
   - Observe processed frames and the model's token/word predictions for lip reading.

4. **Further Development**
   - Experiment with different model architectures or datasets for improved accuracy.
   - Enhance functionalities or add new features to the application.

## Contributing
Currently I am working to generate translations for live video through the web cam and training the model for different languages and multiple speaker simultaneously, Contributions, issues, and feature requests are welcome!
