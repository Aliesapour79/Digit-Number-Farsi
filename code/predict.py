import os
import cv2
import numpy as np
from keras.models import load_model

# Load the pretrained neural network model
model = load_model('C:/Users/Laptop/Documents/pycodes/AzHosh/Digit_Number_Farsi/pretrained_model.h5')

# Function for detecting digits
def detect_numbers(image):
    # Image preprocessing
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Prediction using the neural network
    prediction = model.predict(image)
    
    # Predicted label
    predicted_digit = np.argmax(prediction)
    
    # Prediction probability
    probability = np.max(prediction) * 100
    return predicted_digit, probability

# Path to Test and Train directories
test_dir = "H:/Datasets/Handwritten_Digits/Test"

# Read and detect images in the Test directory
for digit_dir in os.listdir(test_dir):
    digit_path = os.path.join(test_dir, digit_dir)
    if os.path.isdir(digit_path):
        digit_number = int(os.path.basename(digit_path))
        for filename in os.listdir(digit_path):
            image_path = os.path.join(digit_path, filename)
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Detect digits in the image
            predicted_digit, probability = detect_numbers(image)
            
            # Display the result
            print('Image:', image_path)
            print('Ground Truth:', digit_number)
            print('Predicted Digit:', predicted_digit)
            print('Probability:', '{:.3f}%'.format(probability))
