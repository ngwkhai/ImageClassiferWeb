import os
import tarfile
import certifi
import urllib.request

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
dataset_dir = "datasets"

os.makedirs(dataset_dir, exist_ok=True)

# Tải bộ dữ liệu CIFAR-10
file_path = os.path.join(dataset_dir, "cifar-10-python.tar.gz")
if not os.path.exists(file_path):
    print("Downloading CIFAR-10 dataset...")
    with urllib.request.urlopen(url, cafile=certifi.where()) as response, open(file_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download complete.")

# Giải nén bộ dữ liệu
if not os.path.exists(os.path.join(dataset_dir, 'cifar-10-batches-py')):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=dataset_dir)
    print("Extraction complete.")
