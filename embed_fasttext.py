import pandas as pd
import numpy as np
from pathlib import Path
import fasttext
import fasttext.util
import os

from gensim.models import FastText  # We'll use gensim's FastText for compatibility

# === Parameters ===
kmer_size = 3
vector_size = 100
window_size = 5
min_count = 1
epochs = 20

# === Paths ===
base_dir = Path(__file__).resolve().parent.parent
train_csv = base_dir / "data" / "processed" / "Trsequences.csv"
test_csv = base_dir / "data" / "processed" / "Tesequences.csv"
model_path = base_dir / "models" / "fasttext.model"
output_train = base_dir / "embeddings" / "fasttext_train.csv"
output_test = base_dir / "embeddings" / "fasttext_test.csv"

output_train.parent.mkdir(parents=True, exist_ok=True)

# === Helper: k-mer tokenizer ===
# def kmer_split(seq, k=3):
#     return [seq[i:i+k] for i in range(len(seq) - k + 1)]
#
# # === Read sequence data ===
# def load_sequences(path):
#     df = pd.read_csv(path)
#     sequences = df["Sequence"].str.upper().tolist()
#     labels = df["Label"].tolist()
#     kmers = [kmer_split(seq, k=kmer_size) for seq in sequences]
#     return kmers, labels

# === Train FastText model ===
print("[INFO] Loading training sequences...")

def kmer_split(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]
def load_sequences(path):
    df = pd.read_csv(path)
    sequences = df["Sequence"].str.upper().tolist()
    labels = df["Label"].tolist()
    kmers = [kmer_split(seq, k=kmer_size) for seq in sequences]
    return kmers, labels
train_kmers, train_labels = load_sequences(train_csv)

print("[INFO] Training FastText model...")
ft_model = FastText(sentences=train_kmers, vector_size=vector_size,
                    window=window_size, min_count=min_count,
                    workers=4,
                    sg=1,  # use skip-gram
                    epochs=epochs)
ft_model.save(str(model_path))
# === Embed function ===
def embed_sequences(kmer_list, labels, model):
    embeddings = []
    for kmers, label in zip(kmer_list, labels):
        vecs = [model.wv[k] for k in kmers if k in model.wv]
        if vecs:
            avg_vec = np.mean(vecs, axis=0)
        else:
            avg_vec = np.zeros(model.vector_size)
        embeddings.append(list(avg_vec) + [label])
    return embeddings

# === Embed and save train set ===
print("[INFO] Embedding training set...")
train_embedded = embed_sequences(train_kmers, train_labels, ft_model)
pd.DataFrame(train_embedded).to_csv(output_train, header=False, index=False)
print(f"[✓] Saved train embeddings to {output_train}")

# === Embed and save test set ===
print("[INFO] Embedding test set...")
test_kmers, test_labels = load_sequences(test_csv)
test_embedded = embed_sequences(test_kmers, test_labels, ft_model)
pd.DataFrame(test_embedded).to_csv(output_test, header=False, index=False)
print(f"[✓] Saved test embeddings to {output_test}")