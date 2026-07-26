import io
import os
import urllib.request

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv


METADATA_URL = (
    "https://raw.githubusercontent.com/mavaali/cricket-mcp/"
    "main/data/player_meta.csv"
)


def build_player_registry():
    load_dotenv()
    with urllib.request.urlopen(METADATA_URL) as response:
        metadata = pd.read_csv(io.BytesIO(response.read()))

    connection = snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA"),
    )
    players = pd.read_sql(
        """
        SELECT PLAYER, MAX(IS_ACTIVE) AS IS_ACTIVE
        FROM (
            SELECT STRIKER AS PLAYER,
                   IFF(TRY_TO_NUMBER(SEASON) = (
                       SELECT MAX(TRY_TO_NUMBER(SEASON))
                       FROM RAW_DELIVERIES
                   ), 1, 0) AS IS_ACTIVE
            FROM RAW_DELIVERIES
            UNION ALL
            SELECT BOWLER AS PLAYER,
                   IFF(TRY_TO_NUMBER(SEASON) = (
                       SELECT MAX(TRY_TO_NUMBER(SEASON))
                       FROM RAW_DELIVERIES
                   ), 1, 0) AS IS_ACTIVE
            FROM RAW_DELIVERIES
        )
        WHERE PLAYER IS NOT NULL
        GROUP BY PLAYER
        """,
        connection,
    )
    connection.close()

    registry = players.merge(
        metadata,
        left_on="PLAYER",
        right_on="unique_name",
        how="left",
    )
    output = pd.DataFrame(
        {
            "PLAYER": registry["PLAYER"],
            "BATTING_HAND": registry["batting_style"].fillna("Unknown"),
            "BOWLING_TYPE": registry["bowling_style"].fillna("Unknown"),
            "PLAYING_ROLE": registry["playing_role"].fillna("Unknown"),
            "COUNTRY": registry["country"].fillna("Unknown"),
            "IS_ACTIVE": registry["IS_ACTIVE"].astype(bool),
        }
    ).sort_values("PLAYER")
    output_path = os.path.join("data", "player_metadata.csv")
    output.to_csv(output_path, index=False)
    matched = (output["BATTING_HAND"] != "Unknown").sum()
    print(
        f"Saved {len(output):,} IPL players to {output_path}; "
        f"{matched:,} have verified batting styles."
    )


if __name__ == "__main__":
    build_player_registry()
