import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_URL = "https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter/resolve/refs%2Fconvert%2Fparquet/pcm"
RAW_DATA_PATH = "../data/raw/pcm_raw.csv"
PROCESSED_DIR = "../data/processed"


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def fetch_and_save_raw_data() -> pd.DataFrame:
    print("Fetching raw datasets...")
    train_df = pd.read_parquet(f"{BASE_URL}/train/0000.parquet")
    dev_df = pd.read_parquet(f"{BASE_URL}/validation/0000.parquet")
    test_df = pd.read_parquet(f"{BASE_URL}/test/0000.parquet")

    df_raw = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset to '{RAW_DATA_PATH}' ({len(df_raw)} rows).")

    return df_raw


def process_and_split_data(df: pd.DataFrame):
    print("Deduplicating and cleaning data...")
    df["clean_tweet"] = df["tweet"].apply(normalize_text)

    df_dedup = df.drop_duplicates(subset=["clean_tweet"], keep="first").copy()
    df_dedup = df_dedup.drop(columns=["clean_tweet"])

    print(f"Total rows after deduplication: {len(df_dedup)}")

    train_df, temp_df = train_test_split(
        df_dedup, test_size=0.30, random_state=45, stratify=df_dedup["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=45, stratify=temp_df["label"]
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print("Successfully saved processed splits to '../data/processed/':")
    print(f"  - Train: {len(train_df)} rows")
    print(f"  - Validation: {len(val_df)} rows")
    print(f"  - Test: {len(test_df)} rows")


def main():
    df_raw = fetch_and_save_raw_data()
    process_and_split_data(df_raw)


if __name__ == "__main__":
    main()
