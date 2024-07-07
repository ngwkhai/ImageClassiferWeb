import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from train_model import load_data

# Load dữ liệu
(x_train, y_train), (x_test, y_test) = load_data()

# Chuẩn hoá dữ liệu
x_test = x_test.astype('float32') / 255.0

# Chuyển đổi thành one-hot encoding
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Load mô hình đã huấn luyện
model = load_model('models/cifar10_model.h5')

# Đánh giá mô hình trên tập dữ liệu kiểm tra
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Đánh giá các chỉ số khác như precision, recall, f1-score
from sklearn.metrics import classification_report

# Dự đoán trên tập dữ liệu kiểm tra
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Tạo báo cáo phân loại
report = classification_report(y_true, y_pred_classes, target_names=[
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

print(report)
