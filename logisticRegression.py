import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from skimage import feature

# ---------------- CONFIG ----------------
DATASET_PATH = "D:/practice/test_set"
IMG_SIZE = (128, 128)

# ---------------- HOG ----------------
def extract_hog(img):
    img = img.astype("float32") / 255.0

    try:
        return feature.hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1,
            feature_vector=True
        )
    except TypeError:
        return feature.hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            multichannel=True,
            feature_vector=True
        )

# ---------------- LOAD IMAGE ----------------
def load_image(path):
    try:
        return np.array(Image.open(path).convert("RGB").resize(IMG_SIZE))
    except:
        return None

# ---------------- DATASET ----------------
def load_dataset(path):
    path = Path(path)
    classes = sorted([d for d in path.iterdir() if d.is_dir()])  # ✅ FIX: sorted

    label_map = {c.name.lower(): i for i, c in enumerate(classes)}
    inv_label_map = {v: k for k, v in label_map.items()}

    X, y = [], []

    for c in classes:
        for img_path in c.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                img = load_image(img_path)
                if img is None:
                    continue
                X.append(extract_hog(img))
                y.append(label_map[c.name.lower()])

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    return (X[:split], y[:split]), (X[split:], y[split:]), label_map, inv_label_map

# ---------------- TRAIN ----------------
def train_model(X, y):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X, y)
    return model

# ---------------- EVALUATE ----------------
def evaluate(model, X, y, name, inv_label_map):
    pred = model.predict(X)

    acc = accuracy_score(y, pred)
    print(f"{name} Accuracy: {acc * 100:.2f}%")

    labels = np.unique(y)
    display_labels = [inv_label_map[i] for i in labels]

    cm = confusion_matrix(y, pred, labels=labels)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels
    )
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ---------------- PREDICT ----------------
def predict_image(path, model, label_map, inv_label_map):
    print("\nPredicting:", path)

    img = load_image(path)
    if img is None:
        print("Error loading image")
        return

    feat = extract_hog(img).reshape(1, -1)

    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]

    label_name = inv_label_map[pred]
    conf = prob[pred]

    print(f"Result: {label_name.upper()} ({conf * 100:.2f}%)")

    # ---------------- IMAGE ----------------
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"{label_name.upper()} ({conf*100:.1f}%)")
    ax.axis("off")

    # ---------------- CONFIDENCE BAR ----------------
    classes = list(inv_label_map.values())
    probs = prob

    plt.figure(figsize=(5, 3))
    plt.bar(classes, probs)
    plt.title("Confidence Scores")
    plt.ylim(0, 1)
    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test), label_map, inv_label_map = load_dataset(DATASET_PATH)

    model = train_model(X_train, y_train)

    print("\nEVALUATION:")
    evaluate(model, X_train, y_train, "Train", inv_label_map)
    evaluate(model, X_test, y_test, "Test", inv_label_map)

    joblib.dump(model, "cat_dog_model.pkl")
    print("\nModel saved.")

    # ---------------- TEST ----------------
    predict_image("D:/practice/test_set/dogs/dog.4001.jpg", model, label_map, inv_label_map)
    predict_image("D:/practice/test_set/cats/cat.4001.jpg", model, label_map, inv_label_map)