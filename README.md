

## README: Sign Language Detection Using Deep Learning

### Overview

This project aims to develop a system for recognizing American Sign Language (ASL) characters using deep learning techniques. The primary objective is to bridge the communication gap between individuals who use sign language and those who do not by converting hand gestures into text and speech.

### Key Components

1. **Data Collection and Preprocessing:**
   - **Image Acquisition:** Real-time images of hand gestures are captured using a webcam.
   - **Image Preprocessing:** Includes converting RGB images to HSV color space, applying Gaussian blur for noise reduction, and performing edge detection using techniques like Canny edge detection.

2. **Feature Extraction:**
   - **HSV Masking:** Used for color segmentation and contour extraction of hand images.
   - **Clustering:** K-means clustering is applied to enhance image segmentation.

3. **Model Building:**
   - **CNN (Convolutional Neural Network):** Used for image classification tasks.
   - **VGG16:** A deep neural network known for its accuracy in image recognition tasks.
   - **ResNet50:** Another deep learning model that overcomes the vanishing gradient problem through skip connections.
   - **YOLOv5:** Used for real-time object detection and feature extraction.

4. **Training and Testing:**
   - **Training:** 70% of the dataset is used for training the models. The models are trained using TensorFlow and involve various data augmentation techniques.
   - **Testing:** 30% of the dataset is reserved for testing to evaluate the performance of the models.

5. **Real-Time Detection:**
   - **Sign to Text:** TensorFlow is used to convert sign language gestures to text.
   - **Text to Speech:** gTTS (Google Text to Speech) API is used to convert text into speech.

### System Architecture

The system involves capturing hand gestures in real-time, preprocessing the images, extracting features, and using trained models to classify the gestures into corresponding alphabets or words. The classified text can then be converted into speech.

### Results

- **Accuracy and Loss Evaluation:** The models (CNN, VGG16, ResNet50) showed increasing accuracy and decreasing loss with more epochs.
- **Confusion Matrix:** Demonstrated the effectiveness of the models in correctly classifying the ASL characters.

### Conclusion

The project successfully developed a system capable of recognizing and translating ASL gestures into text and speech. This technology can significantly aid in communication for the hearing and speech impaired.

### References

A comprehensive list of references used in the project, including research papers on sign language recognition and deep learning techniques.
