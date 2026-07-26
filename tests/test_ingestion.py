import pandas as pd

from src.ingestion import (
    RAW_TABLE_DDL,
    filter_unseen_match_files,
    prepare_raw_deliveries,
)


def test_wide_no_ball_and_following_legal_ball_are_preserved():
    deliveries = pd.DataFrame(
        {
            "MATCH_ID": ["match-1", "match-1", "match-1"],
            "SEASON": ["2026", "2026", "2026"],
            "INNINGS": [1, 1, 1],
            "BALL": ["0.1", "0.2", "0.3"],
            "ACTUAL_DELIVERY": ["0.1", "0.1", "0.1"],
            "DELIVERY_SEQ": [0, 1, 2],
            "WIDES": [1, 0, 0],
            "NOBALLS": [0, 1, 0],
        }
    )

    prepared = prepare_raw_deliveries([deliveries])

    assert prepared["BALL"].tolist() == ["0.1", "0.2", "0.3"]
    assert prepared["ACTUAL_DELIVERY"].tolist() == ["0.1", "0.1", "0.1"]
    assert len(prepared) == 3


def test_string_delivery_identifiers_do_not_collapse_decimal_suffixes():
    deliveries = pd.DataFrame(
        {
            "MATCH_ID": ["match-1", "match-1"],
            "SEASON": ["2026", "2026"],
            "INNINGS": [1, 1],
            "BALL": ["23.1", "23.10"],
            "ACTUAL_DELIVERY": ["23.1", "23.9"],
            "DELIVERY_SEQ": [0, 1],
        }
    )

    prepared = prepare_raw_deliveries([deliveries])

    assert prepared["BALL"].tolist() == ["23.1", "23.10"]
    assert len(prepared) == 2


def test_true_duplicate_delivery_identifier_is_removed():
    deliveries = pd.DataFrame(
        {
            "MATCH_ID": ["match-1", "match-1"],
            "SEASON": ["2026", "2026"],
            "INNINGS": [1, 1],
            "BALL": ["4.3", "4.3"],
            "ACTUAL_DELIVERY": ["4.3", "4.3"],
            "DELIVERY_SEQ": [10, 11],
        }
    )

    prepared = prepare_raw_deliveries([deliveries])

    assert len(prepared) == 1
    assert prepared.iloc[0]["DELIVERY_SEQ"] == 10


def test_incremental_ingestion_keeps_only_unseen_match_files():
    files = [
        "1001.csv",
        "1002.csv",
        "nested/1003.csv",
    ]

    unseen = filter_unseen_match_files(files, {"1001", "1003"})

    assert unseen == ["1002.csv"]


def test_raw_table_is_created_without_replacing_existing_history():
    assert "CREATE TABLE IF NOT EXISTS RAW_DELIVERIES" in RAW_TABLE_DDL
    assert "CREATE OR REPLACE" not in RAW_TABLE_DDL
