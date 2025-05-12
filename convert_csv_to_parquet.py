import pandas as pd
import sys


def csv_to_parquet(csv_path, parquet_path):
    df = pd.read_csv(csv_path, header=None, names=["text"])
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_parquet.py input.csv output.parquet")
        sys.exit(1)
    csv_to_parquet(sys.argv[1], sys.argv[2])
