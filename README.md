# Handwritten Farsi Digits Classification

This is a simple program for recognizing handwritten Farsi digits using a pretrained convolutional neural network. The program consists of two parts: training the model and using it for prediction.

## Model Training

The model is trained using a dataset of handwritten Farsi digits. The training script (`train_model.py`) uses the Keras library with a TensorFlow backend. The architecture of the model includes multiple convolutional layers followed by max pooling and fully connected layers. The training script saves the trained model to a file (`pretrained_model.h5`).

### Usage

1. Set the paths to the training and validation datasets in the `train_data_dir` and `validation_data_dir` variables in `train_model.py`.
2. Adjust other configurations such as image dimensions, batch size, and number of classes if needed.
3. Run the script using a Python environment with the necessary dependencies installed.

## Model Prediction

The pretrained model is loaded using the `load_model` function from Keras. The prediction script (`predict_numbers.py`) reads images from a test directory, preprocesses them, and uses the model to predict the handwritten digits.

### Usage

1. Set the path to the test dataset in the `test_dir` variable in `predict_numbers.py`.
2. Run the script using a Python environment with the necessary dependencies installed.

## Dependencies

- Python 3.x
- Keras
- TensorFlow
- Matplotlib
- OpenCV

## Example

To see the program in action, run the training script to train the model and then run the prediction script to see the model's predictions on the test dataset.
