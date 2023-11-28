import os
import numpy as np
from tensorflow.keras.preprocessing import image as img
from tensorflow.keras.utils import to_categorical


def load_data(data_dir):
    x_train = []
    y_train = []
    num_class = len(os.listdir(data_dir))
    for i in range(num_class):
        class_dir = os.path.join(data_dir, str(i))
        image_filenames = os.listdir(class_dir)

        for image_filename in image_filenames:
            image_path = os.path.join(class_dir, image_filename)
            image = img.load_img(image_path, target_size=(384, 384))
            image = img.img_to_array(image, dtype=np.float16)

            label = np.zeros((384, 384, 1))
            label[:, :, 0] = i / 9.0  # نرمال‌سازی برچسب به بازه 0 تا 1
            label = to_categorical(label, num_classes=21)  # تبدیل به one-hot encoding
            x_train.append(image)
            y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    return x_train , y_train