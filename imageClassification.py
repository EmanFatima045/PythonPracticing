import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

# sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# keras (only for feature extraction — no training)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

print("TensorFlow:", tf.__version__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "D:/practice/test_set"   # unlabelled images anywhere inside
IMG_SIZE     = (160, 160)
N_CLUSTERS   = 3                         # cat | dog | other
PCA_COMPONENTS = 128                     # reduce 1280-d MobileNet features
CONFIDENCE_THRESHOLD = 0.60             # min distance ratio to call "confident"

# ImageNet class-index ranges (for auto-labelling clusters, not training)
CAT_INDICES = set(range(281, 286))
DOG_INDICES = set(range(151, 269))

# ─────────────────────────────────────────────
# 1. LOAD FEATURE EXTRACTOR  (no top, no training)
# ─────────────────────────────────────────────
print("\n[1/5] Loading MobileNetV2 feature extractor...")
feat_extractor = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'          # → 1280-d vector per image
)
feat_extractor.trainable = False

print("[1/5] Loading full ImageNet classifier (for cluster labelling)...")
imagenet_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=True,
    weights='imagenet'
)
print("      Done ✅")

# ─────────────────────────────────────────────
# 2. COLLECT ALL IMAGES (no labels needed)
# ─────────────────────────────────────────────
print("\n[2/5] Scanning dataset for images (no labels used)...")

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}

def collect_images(root: str):
    paths = []
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
    return sorted(paths)

all_paths = collect_images(DATASET_PATH)
print(f"      Found {len(all_paths)} images")
assert len(all_paths) >= N_CLUSTERS, "Need at least as many images as clusters."

# ─────────────────────────────────────────────
# 3. EXTRACT FEATURES WITH MOBILENETV2
# ─────────────────────────────────────────────
print(f"\n[3/5] Extracting features ({len(all_paths)} images)...")

def extract_feature(img_path: str) -> np.ndarray:
    img   = load_img(img_path, target_size=IMG_SIZE)
    arr   = img_to_array(img)
    arr   = preprocess_input(arr)
    arr   = np.expand_dims(arr, 0)
    feat  = feat_extractor.predict(arr, verbose=0)
    return feat[0]   # shape (1280,)

features   = []
valid_paths = []
for i, path in enumerate(all_paths):
    try:
        f = extract_feature(path)
        features.append(f)
        valid_paths.append(path)
    except Exception as e:
        print(f"      Skipping {path}: {e}")
    if (i + 1) % 100 == 0:
        print(f"      {i+1}/{len(all_paths)} done...")

features = np.array(features)   # (N, 1280)
print(f"      Feature matrix: {features.shape}")

# ─────────────────────────────────────────────
# 4. PCA  →  KMeans CLUSTERING  (sklearn only)
# ─────────────────────────────────────────────
print(f"\n[4/5] PCA ({PCA_COMPONENTS} components) + KMeans ({N_CLUSTERS} clusters)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_.sum() * 100
print(f"      PCA explains {explained:.1f}% of variance")

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)   # (N,)  values: 0,1,2

sil = silhouette_score(X_pca, cluster_labels, sample_size=min(2000, len(X_pca)))
print(f"      Silhouette score: {sil:.3f}  (higher is better, max=1.0)")

# ── Auto-label each cluster using ImageNet on ~30 samples per cluster ──────
print("      Auto-labelling clusters via ImageNet...")

CLUSTER_NAMES = {}   # cluster_id → 'cat' | 'dog' | 'other'

for cid in range(N_CLUSTERS):
    idxs    = np.where(cluster_labels == cid)[0]
    sample  = np.random.choice(idxs, size=min(30, len(idxs)), replace=False)
    cat_votes, dog_votes = 0, 0

    for idx in sample:
        path = valid_paths[idx]
        try:
            img  = load_img(path, target_size=IMG_SIZE)
            arr  = img_to_array(img)
            arr  = preprocess_input(arr)
            arr  = np.expand_dims(arr, 0)
            pred = imagenet_model.predict(arr, verbose=0)
            top  = int(np.argmax(pred[0]))
            if top in CAT_INDICES:
                cat_votes += 1
            elif top in DOG_INDICES:
                dog_votes += 1
        except:
            pass

    total = len(sample)
    if cat_votes / total >= 0.4:
        CLUSTER_NAMES[cid] = 'cat'
    elif dog_votes / total >= 0.4:
        CLUSTER_NAMES[cid] = 'dog'
    else:
        CLUSTER_NAMES[cid] = 'other'

    print(f"      Cluster {cid}: cat={cat_votes}  dog={dog_votes}  → '{CLUSTER_NAMES[cid]}'")

# guard: if two clusters got same name, keep the one with more votes, mark other 'other'
seen = {}
for cid, name in CLUSTER_NAMES.items():
    if name in seen:
        # keep whichever cluster is larger
        prev_cid = seen[name]
        if np.sum(cluster_labels == cid) > np.sum(cluster_labels == prev_cid):
            CLUSTER_NAMES[prev_cid] = 'other'
            seen[name] = cid
        else:
            CLUSTER_NAMES[cid] = 'other'
    else:
        seen[name] = cid

print(f"      Final cluster map: {CLUSTER_NAMES}")

# ─────────────────────────────────────────────
# 5.  VISUALISE CLUSTERS (PCA 2-D projection)
# ─────────────────────────────────────────────
print("\n[5/5] Plotting cluster visualisation...")

COLOR_MAP = {'cat': '#4ECDC4', 'dog': '#FF6B35', 'other': '#95A5A6'}
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
for cid in range(N_CLUSTERS):
    mask  = cluster_labels == cid
    name  = CLUSTER_NAMES[cid]
    color = COLOR_MAP[name]
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=color, label=f"Cluster {cid}: {name} ({mask.sum()})",
               alpha=0.55, s=18, edgecolors='none')

ax.set_title("Unsupervised Clustering — Cat / Dog / Other\n(PCA 2-D projection)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("cluster_visualisation.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_image(img_path: str):
    """
    Unsupervised prediction:
      1. Extract 1280-d MobileNet feature.
      2. Scale + PCA (same fitted transforms).
      3. Find nearest KMeans cluster centroid.
      4. Map cluster → cat / dog / other.
      5. Use distance ratio as a confidence proxy.
    """
    print("\n" + "="*55)
    print("  🔍 PREDICTION (Unsupervised)")
    print("="*55)
    print(f"  Image : {img_path}")

    # — feature extraction —
    raw_feat = extract_feature(img_path)                      # (1280,)
    x_s      = scaler.transform(raw_feat.reshape(1, -1))      # (1, 1280)
    x_p      = pca.transform(x_s)                             # (1, 128)

    # — distances to all centroids —
    dists    = np.linalg.norm(kmeans.cluster_centers_ - x_p, axis=1)  # (K,)
    nearest  = int(np.argmin(dists))
    label    = CLUSTER_NAMES[nearest]

    # — confidence proxy: how much closer is nearest vs second-nearest —
    sorted_dists = np.sort(dists)
    ratio        = 1.0 - (sorted_dists[0] / (sorted_dists[1] + 1e-9))
    confidence   = float(np.clip(ratio, 0, 1)) * 100

    COLOR_LABELS = {
        'cat':   ('🐱 CAT',           '#4ECDC4'),
        'dog':   ('🐶 DOG',           '#FF6B35'),
        'other': ('❓ Other Animal',  '#95A5A6'),
    }
    disp_label, color = COLOR_LABELS[label]

    print(f"  Nearest cluster  : {nearest}  ({label})")
    print(f"  Distances        : { {i: f'{d:.2f}' for i, d in enumerate(dists)} }")
    print(f"  Confidence proxy : {confidence:.1f}%")
    print(f"  Result           : {disp_label}")
    print("="*55)

    # — bar data for all clusters —
    total_dist = dists.sum()
    bar_vals   = {CLUSTER_NAMES[i]: (1 - dists[i] / total_dist) for i in range(N_CLUSTERS)}

    # — plot —
    display_img = load_img(img_path, target_size=IMG_SIZE)
    fig, axes   = plt.subplots(1, 2, figsize=(10, 5),
                               gridspec_kw={'width_ratios': [1, 1]})
    fig.patch.set_facecolor('#1A1A2E')

    # image panel
    axes[0].imshow(display_img)
    axes[0].axis('off')
    axes[0].set_title(f"{disp_label}\nConfidence: {confidence:.1f}%",
                      fontsize=13, fontweight='bold', color=color, pad=12)

    # bar panel
    species  = ['cat', 'dog', 'other']
    vals     = [bar_vals.get(s, 0) for s in species]
    colors   = [COLOR_MAP[s] for s in species]
    bars     = axes[1].barh(species, vals, color=colors, height=0.5, edgecolor='none')
    axes[1].set_xlim(0, max(vals) * 1.25)
    axes[1].set_title("Cluster Proximity Scores",
                      color='white', fontsize=11, pad=10)
    axes[1].tick_params(colors='white')
    axes[1].set_facecolor('#1A1A2E')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#444')
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{v:.2f}', va='center', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight',
                facecolor='#1A1A2E')
    plt.show()

    return label, confidence


# ─────────────────────────────────────────────
# TEST PREDICTIONS
# ─────────────────────────────────────────────
predict_image("D:/practice/test_set/dogs/dog.4001.jpg")
predict_image("D:/practice/test_set/cats/cat.4003.jpg")
predict_image("D:/practice/animal.avif")
predict_image("D:/practice/unknown.jpg")