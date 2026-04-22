# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 2. DATASET PATH
# ===============================
dataset_path = "D:/practice/test_set"

# ===============================
# 3. LOAD DATASET
# ===============================
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(150, 150),
    batch_size=32,
    shuffle=True
)

class_names = dataset.class_names
print("Classes:", class_names)

# ===============================
# 4. NORMALIZE
# ===============================
dataset = dataset.map(lambda x, y: (x / 255.0, y))

# ===============================
# 5. SPLIT DATA
# ===============================
train_size = int(0.8 * len(dataset))

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)

# ===============================
# 6. BUILD MODEL (FAST)
# ===============================
model = models.Sequential([
    layers.Input(shape=(150,150,3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# ===============================
# 7. COMPILE
# ===============================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 8. TRAIN
# ===============================
history = model.fit(train_ds, epochs=5)

# ===============================
# 9. TEST
# ===============================
loss, acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# ===============================
# 10. PREDICTION FUNCTION
# ===============================
def predict_image(img_path):
    print("\n==============================")
    print("FINAL PREDICTION")
    print("==============================")

    img = load_img(img_path, target_size=(150,150))
    
    plt.imshow(img)
    plt.axis('off')

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = f"DOG 🐶 ({prediction*100:.2f}%)"
    else:
        label = f"CAT 🐱 ({(1-prediction)*100:.2f}%)"

    print(label)
    plt.title(label)

# ===============================
# 11. CALL PREDICTION (IMPORTANT)
# ===============================
predict_image("D:/practice/test_set/cats/cat.4001.jpg")

# ===============================
# 12. SHOW GRAPH (AFTER PREDICTION)
# ===============================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title("Model Accuracy")
plt.legend()

plt.show()