import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def read_fasta(file_path, label):
    """Reads a FASTA file and assigns a label to each sequence."""
    sequences = []
    with open(file_path, 'r') as f:
        seq_id, seq = None, ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id and seq:
                    sequences.append([seq_id, seq, label])
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id and seq:
            sequences.append([seq_id, seq, label])
    return sequences

# === Get base directory (project root) ===
BASE_DIR = Path(__file__).resolve().parent.parent

# === File Paths ===
pos_path = BASE_DIR / "data" / "raw" / "PDB14189_P.txt"
neg_path = BASE_DIR / "data" / "raw" / "PDB14189_N.txt"
combined_csv = BASE_DIR / "data" / "processed" / "PNsequences.csv"
train_csv = BASE_DIR / "data" / "processed" / "Trsequences.csv"
test_csv = BASE_DIR / "data" / "processed" / "Tesequences.csv"

# === Read and combine ===
pos_seqs = read_fasta(pos_path, label=1)
neg_seqs = read_fasta(neg_path, label=0)
all_seqs = pos_seqs + neg_seqs

df = pd.DataFrame(all_seqs, columns=["ID", "Sequence", "Label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Split 70/30 while preserving class balance ===
train_df, test_df = train_test_split(df, test_size=0.30, stratify=df["Label"], random_state=42)

# === Save to CSV ===
combined_csv.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
df.to_csv(combined_csv, index=False)
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"[✓] Combined dataset saved to: {combined_csv}")
print(f"[✓] Training set saved to:     {train_csv}")
print(f"[✓] Testing set saved to:      {test_csv}")