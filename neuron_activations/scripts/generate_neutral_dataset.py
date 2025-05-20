import pandas as pd
from datasets import load_dataset
import re
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Download the dataset (train split) in streaming mode
dataset = load_dataset("stas/openwebtext-10k", split="train", streaming=True)


# Helper to extract first sentence (naive split on period, exclamation, or question mark)
def extract_first_sentence(text):
    match = re.search(r"(.+?[.!?])\s", text + " ")
    if match:
        return match.group(1).strip()
    return text.strip().split("\n")[0]  # fallback: first line


# Prepare Arrow writer
out_path = "openwebtext_first_sentences.parquet"
schema = pa.schema([("text", pa.string())])
writer = None

batch_size = 1000
batch = []

for example in tqdm(dataset, desc="Extracting first sentences"):
    text = example.get("text", "")
    if text:
        first = extract_first_sentence(text)
        if first:
            batch.append({"text": first})
    if len(batch) >= batch_size:
        table = pa.Table.from_pandas(
            pd.DataFrame(batch), schema=schema, preserve_index=False
        )
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema)
        writer.write_table(table)
        batch = []

# Write any remaining rows
if batch:
    table = pa.Table.from_pandas(
        pd.DataFrame(batch), schema=schema, preserve_index=False
    )
    if writer is None:
        writer = pq.ParquetWriter(out_path, schema)
    writer.write_table(table)

if writer is not None:
    writer.close()
    print(f"Saved first sentences to {out_path}")
else:
    print("No data extracted.")
