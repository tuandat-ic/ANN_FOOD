!pip install tensorflow keras
# 1. Upload & unzip dataset =====
from google.colab import files
import zipfile, os

uploaded = files.upload()  # chọn file .zip (vd: data_mono.zip)
zip_path = next(iter(uploaded.keys()))

extract_path = "/content/F_mono"
!rm -rf {extract_path}
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# 2. Duyệt dữ liệu ảnh =====
import cv2
import numpy as np

x_data, y_data = [], []
# Correctly list the class directories within the nested folder
class_base_path = os.path.join(extract_path, "F_mono")
class_names = sorted([name for name in os.listdir(class_base_path) if os.path.isdir(os.path.join(class_base_path, name))])

print("Classes:", class_names)

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(class_base_path, class_name)
# Recursively walk through subdirectories to find image files
    for root, dirs, files_in_dir in os.walk(class_dir):
        for fname in files_in_dir:
            fpath = os.path.join(root, fname)
            # Check if the file is an image (basic check based on extension)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image file: {fpath}")
                    continue
                img_resized = cv2.resize(img, (28, 28))
                x_data.append(img_resized)
                y_data.append(label)

x_data = np.array(x_data, dtype="float32") / 255.0
y_data = np.array(y_data, dtype="int")

print("Dataset shape:", x_data.shape, y_data.shape)

  # 3. Reshape & one-hot
if x_data.shape[0] > 0: # Only proceed if data is loaded
  x_data = x_data.reshape(x_data.shape[0], 28*28)  # flatten
  from keras.utils import to_categorical
  y_data = to_categorical(y_data, num_classes=len(class_names))

  # 4. Chia train/test
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(
      x_data, y_data, test_size=0.2, random_state=42 # Removed stratify=y_data
  )

  print("Train:", x_train.shape, y_train.shape)
  print("Test:", x_test.shape, y_test.shape)

  #5. Xây dựng ANN
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.layers import Dropout # Import Dropout layer

  model = Sequential()
  model.add(Dense(512, activation='relu', input_shape=(784,)))
  model.add(Dropout(0.3)) # Added Dropout layer
  model.add(Dense(256, activation='relu')) # Added Dense layer
  model.add(Dropout(0.3)) # Added Dropout layer
  model.add(Dense(len(class_names), activation='softmax')) # Output layer with correct number of classes

  model.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

  model.summary()

  # 6. Huấn luyện
  history = model.fit(x_train, y_train,
                      epochs=200,
                      batch_size=128,
                      validation_data=(x_test, y_test))

  # 7. Lưu model
  model.save("final_model.h5")
  print("✅ Model saved!")
  # 8.Up flie kiểm tra
from google.colab import files
uploaded = files.upload()
fname = next(iter(uploaded.keys()))

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (28,28))
img_norm = img_resized.astype("float32")/255.0
img_ready = img_norm.reshape(1, 28*28)

preds = model.predict(img_ready)
digit = class_names[np.argmax(preds)]
print("Dự đoán:", digit)

plt.imshow(img_norm, cmap="gray")
plt.title(f"DỰ ĐOÁN ĐÂY LÀ : {digit}")
plt.axis("off")
plt.show()
from google.colab import files, output
import cv2, numpy as np
import matplotlib.pyplot as plt

# Hàm dự đoán top 3
def predict_food():
    uploaded = files.upload()  # mở file picker Colab
    fname = next(iter(uploaded.keys()))

    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28,28))
    img_norm = img_resized.astype("float32")/255.0
    img_ready = img_norm.reshape(1, 28*28)

    preds = model.predict(img_ready)[0]
    top_indices = preds.argsort()[-3:][::-1]
    top_labels = [class_names[i] for i in top_indices]
    top_probs = [preds[i] for i in top_indices]

    html_result = f"<div>Dự đoán chính: <b>{top_labels[0]}</b></div>"
    html_top3 = "<ul style='list-style:none; padding-left:0;'>"
    for label, prob in zip(top_labels, top_probs):
        html_top3 += f"<li>{label}: {prob*100:.2f}%</li>"
    html_top3 += "</ul>"

    display(HTML(f"""
        <script>
            document.getElementById('result').innerHTML = `{html_result}`;
            document.getElementById('top3').innerHTML = `{html_top3}`;
            document.getElementById('resetBtn').style.display = 'inline-block';
        </script>
    """))

    plt.imshow(img_norm, cmap="gray")
    plt.title(f"Dự đoán chính: {top_labels[0]}")
    plt.axis("off")
    plt.show()

output.register_callback('predict_food', predict_food)
from IPython.display import display, HTML

display(HTML("""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Dự đoán món ăn Việt Nam 🍜</title>
<link href="https://fonts.googleapis.com/css2?family=Pacifico&family=Nunito:wght@400;700&display=swap" rel="stylesheet">
<style>
    body {
        font-family: 'Nunito', sans-serif;
        background: linear-gradient(135deg, #fffde7 0%, #ffe0b2 100%);
        text-align: center;
        padding: 40px;
    }
    h1 {
        font-family: 'Pacifico', cursive;
        color: #D84315;
        font-size: 50px;
        margin-bottom: 25px;
        text-shadow: 2px 2px #ffccbc;
    }
    .card {
        background: #fffaf0;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.15);
        max-width: 600px;
        margin: auto;
    }
    button {
        margin-top: 20px;
        padding: 14px 35px;
        font-size: 18px;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
        color: white;
    }
    #predictBtn {
        background: linear-gradient(45deg, #FF9800, #F57C00);
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    #predictBtn:hover { transform: scale(1.05); }
    #resetBtn {
        background: linear-gradient(45deg, #43A047, #66BB6A);
        display: none;
    }
    #result {
        margin-top: 25px;
        font-size: 24px;
        font-weight: bold;
        color: #BF360C;
    }
    #top3 {
        margin-top: 15px;
        font-size: 18px;
        color: #4E342E;
        list-style: none;
        padding-left: 0;
    }
    #top3 li {
        padding: 5px 0;
    }
</style>
</head>
<body>
    <h1>🍲 Nhận diện món ăn Việt Nam 🍲</h1>
    <div class="card">
        <button id="predictBtn" onclick="google.colab.kernel.invokeFunction('predict_food', [], {});">
            📸 Chọn ảnh & Dự đoán
        </button>
        <button id="resetBtn" onclick="location.reload();">
            🔄 Dự đoán món khác
        </button>
        <div id="result"></div>
        <ul id="top3"></ul>
    </div>
</body>
</html>
"""))
