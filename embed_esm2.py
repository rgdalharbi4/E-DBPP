import pandas as pd
import numpy as np
import torch
import esm
from pathlib import Path

# === Setup ===
BATCH_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
base_dir = Path(__file__).resolve().parent.parent
train_csv = base_dir / "data" / "processed" / "Trsequences.csv"
test_csv = base_dir / "data" / "processed" / "Tesequences.csv"
output_train = base_dir / "embeddings" / "esm2_train.csv"
output_test = base_dir / "embeddings" / "esm2_test.csv"

output_train.parent.mkdir(parents=True, exist_ok=True)

# === Load ESM model ===
print("[INFO] Loading ESM2 model (T6-8M)...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = model.to(device)
model.eval()
batch_converter = alphabet.get_batch_converter()
def embed_sequences(df, output_path):
    all_embeddings = []
    sequences = df["Sequence"].str.upper().tolist()
    labels = df["Label"].tolist()
    ids = df["ID"].tolist() if "ID" in df.columns else [f"seq{i}" for i in range(len(df))]
    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_data = list(zip(batch_ids, batch_seqs))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_reps = results["representations"][6]
        for j, (_, seq) in enumerate(batch_data):
            seq_len = len(seq)
            emb = token_reps[j, 1:seq_len+1].mean(0).cpu().numpy()
            all_embeddings.append(list(emb) + [labels[i + j]])
        print(f"[✓] Processed {min(i + BATCH_SIZE, len(sequences))}/{len(sequences)} sequences")

    df_out = pd.DataFrame(all_embeddings)
    df_out.to_csv(output_path, index=False, header=False)
    print(f"[✓] Saved to {output_path}")

# === Run on training set ===
print("[INFO] Embedding training set...")
df_train = pd.read_csv(train_csv)
embed_sequences(df_train, output_train)

# === Run on test set ===
print("[INFO] Embedding test set...")
df_test = pd.read_csv(test_csv)
embed_sequences(df_test, output_test)