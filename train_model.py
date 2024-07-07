import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
import pickle

dataset_dir = "datasets/cifar-10-batches-py"

def load_batch(fpath):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        d_decoded = {k.decode('utf8'): v for k, v in d.items()}
    data = d_decoded["data"]
    labels = d_decoded["labels"]
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels

def load_data():
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 32, 32, 3), dtype='uint8')
    y_train = np.empty(num_train_samples, dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dataset_dir, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(dataset_dir, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    return (x_train, y_train), (x_test, y_test)

def main():
    # Load dữ liệu
    (x_train, y_train), (x_test, y_test) = load_data()

    # Chuẩn hoá dữ liệu
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Chuyển đổi thành one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Xây dựng mô hình CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Biên dịch mô hình
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    # Lưu trữ mô hình đã huấn luyện
    model.save('models/cifar10_model.h5')

if __name__ == "__main__":
    main()
