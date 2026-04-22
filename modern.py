
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import load_img, img_to_array

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

DATASET_PATH         = "D:/practice/test_set"
IMG_SIZE             = (160, 160)
BATCH_SIZE           = 32
EPOCHS_FROZEN        = 5
EPOCHS_FINETUNE      = 5
CONFIDENCE_THRESHOLD = 0.85

# ImageNet class index ranges for cats and dogs
CAT_INDICES = set(range(281, 286))   # tabby, Persian, Siamese, etc.
DOG_INDICES = set(range(151, 269))   # all dog breeds

full_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

class_names = full_dataset.class_names
print(f"\nDetected Classes: {class_names}")
assert len(class_names) == 2, "Dataset must have exactly 2 classes: cats/ and dogs/"

total_batches = len(full_dataset)
train_size    = int(0.80 * total_batches)
val_size      = int(0.10 * total_batches)

train_ds = full_dataset.take(train_size)
val_ds   = full_dataset.skip(train_size).take(val_size)
test_ds  = full_dataset.skip(train_size + val_size)

print(f"Batches → Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomBrightness(0.10),
], name="data_augmentation")

def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)
    return images, labels

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
 
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(128, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs, name="CatDog_MobileNetV2")
model.summary()

print("\nLoading ImageNet model for unknown detection...")
imagenet_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=True,       # full 1000-class ImageNet classifier
    weights='imagenet'
)
print("ImageNet model ready ✅")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3, restore_best_weights=True
)

print("\n" + "="*55)
print("  PHASE 1: Training classifier head (base frozen)")
print("="*55)
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FROZEN,
    callbacks=[early_stop]
)

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*55)
print("  PHASE 2: Fine-tuning top layers of MobileNetV2")
print("="*55)
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=[early_stop]
)

loss, acc = model.evaluate(test_ds)  # type: ignore
print(f"\n{'='*55}")
print(f"  Final Test Accuracy : {acc*100:.2f}%")
print(f"  Final Test Loss     : {loss:.4f}")
print(f"{'='*55}")

def plot_history(h1, h2):
    acc1  = h1.history['accuracy']
    acc2  = h2.history['accuracy']
    vacc1 = h1.history['val_accuracy']
    vacc2 = h2.history['val_accuracy']

    all_acc  = acc1  + acc2
    all_vacc = vacc1 + vacc2
    epochs   = range(1, len(all_acc) + 1)
    split    = len(acc1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History — Cat vs Dog Classifier", fontsize=14, fontweight='bold')

    axes[0].plot(epochs, all_acc,  'b-o', label='Train Accuracy',     markersize=4)
    axes[0].plot(epochs, all_vacc, 'r-o', label='Validation Accuracy', markersize=4)
    axes[0].axvline(split + 0.5, color='gray', linestyle='--', label='Fine-tune start')
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    loss1  = h1.history['loss']
    loss2  = h2.history['loss']
    vloss1 = h1.history['val_loss']
    vloss2 = h2.history['val_loss']
    all_loss  = loss1  + loss2
    all_vloss = vloss1 + vloss2

    axes[1].plot(epochs, all_loss,  'b-o', label='Train Loss',      markersize=4)
    axes[1].plot(epochs, all_vloss, 'r-o', label='Validation Loss', markersize=4)
    axes[1].axvline(split + 0.5, color='gray', linestyle='--', label='Fine-tune start')
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()

plot_history(history1, history2)
def predict_image(img_path, threshold=CONFIDENCE_THRESHOLD):
    """
    STEP 1: ImageNet model checks if image is cat or dog at all.
            If not → immediately returns Unknown. (catches elephants, birds, etc.)
    STEP 2: Only if it IS a cat/dog → run your trained classifier.
    """
    print("\n" + "="*55)
    print("  🔍 PREDICTION")
    print("="*55)
    print(f"  Image : {img_path}")

    img       = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    imagenet_preds = imagenet_model.predict(img_array, verbose=0)
    top_index      = int(np.argmax(imagenet_preds[0]))
    top_conf       = float(np.max(imagenet_preds[0]))

    is_cat = top_index in CAT_INDICES
    is_dog = top_index in DOG_INDICES

    print(f"  ImageNet class index : {top_index}  |  Confidence: {top_conf*100:.1f}%")
    print(f"  Is Cat: {is_cat}  |  Is Dog: {is_dog}")

    if not is_cat and not is_dog:
        label      = "❓  Unknown / Other Animal"
        confidence = top_conf * 100
        color      = "#95A5A6"

        print(f"  Result     : {label}")
        print(f"  Reason     : Not recognized as cat or dog by ImageNet")
        print("="*55)

        display_img = load_img(img_path, target_size=IMG_SIZE)
        fig, ax = plt.subplots(figsize=(5, 5.5))
        ax.imshow(display_img)
        ax.axis('off')
        ax.set_title(
            f"{label}\nThis is not a cat or dog!",
            fontsize=12, fontweight='bold', color=color, pad=10
        )
        plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
        plt.show()
        return label, confidence

    raw_pred = model.predict(img_array, verbose=0)[0][0]
    dog_conf = float(raw_pred)
    cat_conf = 1.0 - dog_conf

    print(f"  Classifier → Dog: {dog_conf*100:.1f}%  |  Cat: {cat_conf*100:.1f}%")

    if dog_conf >= threshold:
        label      = " DOG"
        confidence = dog_conf * 100
        color      = "#FF6B35"
    elif cat_conf >= threshold:
        label      = " CAT"
        confidence = cat_conf * 100
        color      = "#4ECDC4"
    else:
        label      = "❓  Unknown / Other Animal"
        confidence = max(dog_conf, cat_conf) * 100
        color      = "#95A5A6"

    print(f"  Result     : {label}")
    print(f"  Confidence : {confidence:.1f}%")
    print("="*55)

    # Display result
    display_img = load_img(img_path, target_size=IMG_SIZE)
    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.imshow(display_img)
    ax.axis('off')
    ax.set_title(
        f"{label}\nConfidence: {confidence:.1f}%",
        fontsize=13, fontweight='bold', color=color, pad=10
    )

    bar_ax = fig.add_axes([0.15, 0.04, 0.70, 0.04])  # type: ignore
    bar_ax.barh(0,  dog_conf, color='#FF6B35', height=1)
    bar_ax.barh(0, -cat_conf, color='#4ECDC4', height=1)
    bar_ax.axvline( threshold, color='black', linestyle='--', linewidth=1.5)
    bar_ax.axvline(-threshold, color='black', linestyle='--', linewidth=1.5)
    bar_ax.set_xlim(-1, 1)
    bar_ax.axis('off')
    bar_ax.legend(
        handles=[
            mpatches.Patch(color='#4ECDC4', label=f'Cat {cat_conf*100:.0f}%'),
            mpatches.Patch(color='#FF6B35', label=f'Dog {dog_conf*100:.0f}%'),
        ],
        loc='lower center', ncol=2, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -1.5)
    )

    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    return label, confidence


predict_image("D:/practice/test_set/dogs/dog.4001.jpg")   
predict_image("D:/practice/test_set/cats/cat.4003.jpg")   
predict_image("D:/practice/animal.avif")                  
predict_image("D:/practice/unknown.jpg") 