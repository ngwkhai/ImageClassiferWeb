import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load mô hình đã huấn luyện
model = load_model('models/cifar10_model.h5')

# Danh sách các lớp
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Định nghĩa route cho API
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem có file trong yêu cầu không
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axits=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_name(np.argmax(prediction[0]))

        return render_template('index.html', prediction=predicted_class)


# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)
