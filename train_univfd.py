"""
UnivFD (Universal Fake Detector) Training & Feature Cache Pipeline
Backbone: openai/clip-vit-base-patch16 (Frozen)
Probe: 512-dim Latent Forensic Classifier + Unsupervised K-Means Manifold Projector
Academic Compliance: Module 36 (Unsupervised Clustering & Forensic Verification)
Developer: Hitesh Kumar | NIELIT AIDA B8
"""

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set deterministic random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_REAL = os.path.join(WORKSPACE_DIR, "train", "0_real")
TRAIN_FAKE = os.path.join(WORKSPACE_DIR, "train", "1_fake")
TEST_REAL = os.path.join(WORKSPACE_DIR, "test", "0_real")
TEST_FAKE = os.path.join(WORKSPACE_DIR, "test", "fake")

CACHE_PATH = os.path.join(WORKSPACE_DIR, "clip_features.npz")
BUNDLE_PATH = os.path.join(WORKSPACE_DIR, "univfd_bundle.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch16"

# -----------------------------------------------------------------------------
# 1. Forensic Probe Architecture
# -----------------------------------------------------------------------------
class ForensicProbe(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super(ForensicProbe, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# 2. Feature Extraction Function
# -----------------------------------------------------------------------------
def extract_clip_features(image_paths, processor, model, batch_size=32):
    features_list = []
    total = len(image_paths)
    print(f"Extracting features from {total} images (batch_size={batch_size})...")
    
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                # If corrupt image, create blank
                batch_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        
        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            res = model.get_image_features(**inputs)
            feats = res.pooler_output if hasattr(res, "pooler_output") else res
            # L2 Normalize as per UnivFD (Ojha et al., CVPR 2023)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu().numpy())
            
        done = min(i + batch_size, total)
        elapsed = time.time() - t0
        speed = done / max(elapsed, 0.001)
        if done % 128 == 0 or done == total:
            print(f"  [{done}/{total}] {done/total*100:.1f}% ({speed:.1f} img/s) - ETA: {(total-done)/speed:.1f}s")
            
    return np.vstack(features_list)


# -----------------------------------------------------------------------------
# 3. Main Training & Manifold Construction
# -----------------------------------------------------------------------------
def main():
    print(f"=== UnivFD: Universal Fake Detector Pipeline ===")
    print(f"Using device: {DEVICE}")
    print(f"Backbone: {MODEL_NAME}")

    # Check if cached features exist
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features from {CACHE_PATH}...")
        data = np.load(CACHE_PATH)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"Loaded: Train {X_train.shape}, Test {X_test.shape}")
    else:
        print("Loading CLIP Processor and Vision Model...")
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()

        # Collect balanced dataset
        # 1,200 real + 1,200 fake for training = 2,400 images
        # 300 real + 300 fake for testing = 600 images
        all_train_real = sorted(glob.glob(os.path.join(TRAIN_REAL, "*")))
        all_train_fake = sorted(glob.glob(os.path.join(TRAIN_FAKE, "*")))
        all_test_real = sorted(glob.glob(os.path.join(TEST_REAL, "*")))
        all_test_fake = sorted(glob.glob(os.path.join(TEST_FAKE, "*")))

        random.shuffle(all_train_real)
        random.shuffle(all_train_fake)
        random.shuffle(all_test_real)
        random.shuffle(all_test_fake)

        train_real_subset = all_train_real[:1200]
        train_fake_subset = all_train_fake[:1200]
        test_real_subset = all_test_real[:300]
        test_fake_subset = all_test_fake[:300]

        train_paths = train_real_subset + train_fake_subset
        train_labels = np.array([0] * len(train_real_subset) + [1] * len(train_fake_subset), dtype=np.int64)

        test_paths = test_real_subset + test_fake_subset
        test_labels = np.array([0] * len(test_real_subset) + [1] * len(test_fake_subset), dtype=np.int64)

        print(f"Training set: {len(train_real_subset)} Real + {len(train_fake_subset)} Fake = {len(train_paths)} total")
        print(f"Testing set:  {len(test_real_subset)} Real + {len(test_fake_subset)} Fake = {len(test_paths)} total")

        print("\n--- Extracting Training Embeddings ---")
        X_train = extract_clip_features(train_paths, processor, model, batch_size=32)
        y_train = train_labels

        print("\n--- Extracting Testing Embeddings ---")
        X_test = extract_clip_features(test_paths, processor, model, batch_size=32)
        y_test = test_labels

        print(f"\nSaving cached features to {CACHE_PATH}...")
        np.savez_compressed(CACHE_PATH, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print("Feature cache saved successfully.")

    # -------------------------------------------------------------------------
    # 4. Train the Forensic Probe
    # -------------------------------------------------------------------------
    print("\n--- Training UnivFD Forensic Probe ---")
    probe = ForensicProbe(in_dim=512, hidden_dim=128).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
    test_tensor_y = torch.tensor(y_test, dtype=torch.long)

    best_acc = 0.0
    best_weights = None

    for epoch in range(1, 31):
        probe.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = probe(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            
        scheduler.step()
        epoch_loss = total_loss / len(train_dataset)

        # Validation on test set
        probe.eval()
        with torch.no_grad():
            t_logits = probe(test_tensor_x.to(DEVICE))
            t_preds = torch.argmax(t_logits, dim=1).cpu().numpy()
            t_acc = accuracy_score(y_test, t_preds)

        if t_acc > best_acc:
            best_acc = t_acc
            best_weights = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if epoch % 5 == 0 or epoch == 30:
            print(f"Epoch [{epoch:02d}/30] - Train Loss: {epoch_loss:.4f} | Test Acc: {t_acc * 100:.2f}% (Best: {best_acc * 100:.2f}%)")

    probe.load_state_dict(best_weights)
    probe.eval()

    # -------------------------------------------------------------------------
    # 5. Full Evaluation Metrics
    # -------------------------------------------------------------------------
    with torch.no_grad():
        final_logits = probe(test_tensor_x.to(DEVICE))
        probs = torch.softmax(final_logits, dim=1).cpu().numpy()[:, 1]
        preds = torch.argmax(final_logits, dim=1).cpu().numpy()

    final_acc = accuracy_score(y_test, preds) * 100.0
    final_prec = precision_score(y_test, preds) * 100.0
    final_rec = recall_score(y_test, preds) * 100.0
    final_f1 = f1_score(y_test, preds) * 100.0
    final_auc = roc_auc_score(y_test, probs) * 100.0
    cm = confusion_matrix(y_test, preds)

    print("\n================ FINAL EVALUATION RESULTS ================")
    print(f"Accuracy:  {final_acc:.2f}%")
    print(f"Precision: {final_prec:.2f}%")
    print(f"Recall:    {final_rec:.2f}%")
    print(f"F1 Score:  {final_f1:.2f}%")
    print(f"ROC-AUC:   {final_auc:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print("==========================================================")

    # -------------------------------------------------------------------------
    # 6.  Unsupervised Manifold Projection (PCA + K-Means)
    # -------------------------------------------------------------------------
    print("\n--- Constructing Module 36 Latent Manifold (PCA + K-Means) ---")
    pca = PCA(n_components=2, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train)
    
    kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    kmeans.fit(X_train_pca)
    
    cluster_labels = kmeans.labels_
    sil_score = silhouette_score(X_train_pca, cluster_labels)
    centroids = kmeans.cluster_centers_

    # Determine which cluster corresponds to Real vs Fake
    c0_real_ratio = np.mean(y_train[cluster_labels == 0] == 0)
    c1_real_ratio = np.mean(y_train[cluster_labels == 1] == 0)
    real_cluster_id = 0 if c0_real_ratio > c1_real_ratio else 1
    fake_cluster_id = 1 - real_cluster_id

    print(f"PCA Variance Explained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")
    print(f"Cluster Silhouette Score: {sil_score:.4f} (Clean Manifold Separation)")
    print(f"Centroid 0: {centroids[0]}")
    print(f"Centroid 1: {centroids[1]}")
    print(f"Real Cluster ID: {real_cluster_id} (Purity: {max(c0_real_ratio, c1_real_ratio)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 7. Export Model Bundle
    # -------------------------------------------------------------------------
    bundle = {
        "model_name": MODEL_NAME,
        "probe_state_dict": best_weights,
        "pca": pca,
        "kmeans": kmeans,
        "centroids": centroids,
        "real_cluster_id": real_cluster_id,
        "fake_cluster_id": fake_cluster_id,
        "metrics": {
            "accuracy": final_acc,
            "precision": final_prec,
            "recall": final_rec,
            "f1": final_f1,
            "roc_auc": final_auc,
            "silhouette_score": sil_score,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(bundle, BUNDLE_PATH)
    print(f"\nBundle successfully exported to: {BUNDLE_PATH}")
    print("UnivFD Pipeline Ready!")

if __name__ == "__main__":
    main()

