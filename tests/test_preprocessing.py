import pandas as pd
import pytest
from src.preprocessing import (
    normalize_text,
    fetch_and_save_raw_data,
    process_and_split_data,
)


def test_normalized_output():
    inputed_text = "NA way ChAI, WHO   be THAT ONe"
    expected_output = "na way chai, who be that one"
    assert normalize_text(inputed_text) == expected_output


def test_normalized_text_hanndles_empty_and_non_string_input():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""  # type: ignore
    assert normalize_text(747476) == ""  # type: ignore


def test_process_and_split_data_deduplication_and_split_ratios(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr("src.preprocessing.PROCESSED_DIR", str(processed_dir))
    data = {
        "tweet": [
            "man must wack",
            "abeg who get tinubu number",
            "sub for me na",
            "okocha na baller",
            "Ronaldo na goat",
            "i don tire to type",
            "but i must type",
            "e no easy",
            "na jude okoye sing am",
            "e don do",
            "make we rest small",
            "life goes on",
            "wetin be this",
            "stress dey here",
            "i wan change my wardrope",
            "people like free things",
            "true talk jare",
            "make we finish am here",
            "okay na",
        ],
        "label": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 0],
    }

    df = pd.DataFrame(data)
    process_and_split_data(df)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    assert train_path.exists()
    assert val_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    total_output_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_output_rows == 19

    assert len(train_df) == 13
    assert len(val_df) == 3
    assert len(test_df) == 3


def test_fetch_and_save_raw_data(tmp_path, monkeypatch):
    raw_data = tmp_path / "raw" / "pcm_raw.csv"
    monkeypatch.setattr("src.preprocessing.RAW_DATA_PATH", str(raw_data))

    mock_df = pd.DataFrame(
        {"tweet": ["e choke!", "for here plenty", "check am"], "label": [2, 0, 1]}
    )

    def mock_read_parquet(url):
        return mock_df

    monkeypatch.setattr(pd, "read_parquet", mock_read_parquet)

    df_raw = fetch_and_save_raw_data()

    assert len(df_raw) == 9
    assert raw_data.exists()
