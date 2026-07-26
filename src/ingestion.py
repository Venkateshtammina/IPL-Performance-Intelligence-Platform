import os
import urllib.request
import zipfile
import io
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

load_dotenv()


TARGET_COLUMNS = [
    'MATCH_ID', 'SEASON', 'START_DATE', 'VENUE', 'INNINGS', 'BALL',
    'DELIVERY_SEQ', 'ACTUAL_DELIVERY', 'BATTING_TEAM', 'BOWLING_TEAM',
    'STRIKER', 'NON_STRIKER', 'BOWLER', 'RUNS_OFF_BAT', 'EXTRAS',
    'WIDES', 'NOBALLS', 'BYES', 'LEGBYES', 'PENALTY', 'WICKET_TYPE',
    'PLAYER_DISMISSED', 'OTHER_WICKET_TYPE', 'OTHER_PLAYER_DISMISSED',
]

RAW_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS RAW_DELIVERIES (
    MATCH_ID VARCHAR, SEASON VARCHAR, START_DATE VARCHAR, VENUE VARCHAR,
    INNINGS INT, BALL VARCHAR, DELIVERY_SEQ INT, ACTUAL_DELIVERY VARCHAR,
    BATTING_TEAM VARCHAR, BOWLING_TEAM VARCHAR, STRIKER VARCHAR,
    NON_STRIKER VARCHAR, BOWLER VARCHAR, RUNS_OFF_BAT INT, EXTRAS INT,
    WIDES INT, NOBALLS INT, BYES INT, LEGBYES INT, PENALTY INT,
    WICKET_TYPE VARCHAR, PLAYER_DISMISSED VARCHAR,
    OTHER_WICKET_TYPE VARCHAR, OTHER_PLAYER_DISMISSED VARCHAR
)
"""

FEATURE_QUEUE_DDL = """
CREATE TABLE IF NOT EXISTS FEATURE_MATCH_QUEUE (
    MATCH_ID VARCHAR,
    QUEUED_AT TIMESTAMP_NTZ
)
"""


def prepare_raw_deliveries(match_frames):
    final_raw_df = pd.concat(match_frames, ignore_index=True)
    final_raw_df['SEASON'] = final_raw_df['SEASON'].astype(str)
    final_raw_df.columns = final_raw_df.columns.str.upper()

    for column in TARGET_COLUMNS:
        if column not in final_raw_df.columns:
            final_raw_df[column] = None

    final_raw_df['BALL'] = final_raw_df['BALL'].astype(str)
    final_raw_df['ACTUAL_DELIVERY'] = final_raw_df['ACTUAL_DELIVERY'].astype(str)
    final_raw_df = final_raw_df[TARGET_COLUMNS]
    final_raw_df = (
        final_raw_df.sort_values(['MATCH_ID', 'INNINGS', 'DELIVERY_SEQ'])
        .drop_duplicates(
            subset=['MATCH_ID', 'INNINGS', 'BALL'],
            keep='first',
        )
        .reset_index(drop=True)
    )
    if final_raw_df.duplicated(['MATCH_ID', 'INNINGS', 'BALL']).any():
        raise ValueError("Duplicate delivery identifiers remain after ingestion")
    return final_raw_df


def filter_unseen_match_files(csv_files, existing_match_ids):
    existing = {str(match_id) for match_id in existing_match_ids}
    return [
        match_file
        for match_file in csv_files
        if os.path.splitext(os.path.basename(match_file))[0] not in existing
    ]


def upload_raw_data_to_snowflake():
    print("🚀 Starting Enterprise Snowflake Cloud Ingestion...")
    CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
    
    print("📥 Connecting to Cricsheet to extract the historical dataset split...")
    try:
        response = urllib.request.urlopen(CRICSHEET_URL)
        zip_data = zipfile.ZipFile(io.BytesIO(response.read()))
        print("✅ Historical zip bundle loaded successfully into memory buffers.")
    except Exception as e:
        print(f"❌ Network operation failed. Error trace: {e}")
        return

    csv_files = [f for f in zip_data.namelist() if f.endswith('.csv') and not f.endswith('_info.csv')]
    try:
        ctx = snowflake.connector.connect(
            user=os.getenv("SF_USER"),
            password=os.getenv("SF_PASSWORD"),
            account=os.getenv("SF_ACCOUNT"),
            warehouse=os.getenv("SF_WAREHOUSE"),
            database=os.getenv("SF_DATABASE"),
            schema=os.getenv("SF_SCHEMA")
        )
        cursor = ctx.cursor()
        cursor.execute(RAW_TABLE_DDL)
        cursor.execute(FEATURE_QUEUE_DDL)
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
              AND TABLE_NAME = 'ANALYTICAL_MATCHUP_FEATURES'
        """)
        feature_table_exists = int(cursor.fetchone()[0]) > 0
        full_refresh = os.getenv("FULL_REFRESH", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if full_refresh:
            cursor.execute("TRUNCATE TABLE RAW_DELIVERIES")
            cursor.execute("TRUNCATE TABLE FEATURE_MATCH_QUEUE")
            if feature_table_exists:
                cursor.execute("TRUNCATE TABLE ANALYTICAL_MATCHUP_FEATURES")
            existing_match_ids = set()
        else:
            cursor.execute("SELECT DISTINCT MATCH_ID FROM RAW_DELIVERIES")
            existing_match_ids = {
                str(row[0]) for row in cursor.fetchall() if row[0] is not None
            }
            missing_feature_filter = (
                """
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM ANALYTICAL_MATCHUP_FEATURES features
                    WHERE features.MATCH_ID = raw.MATCH_ID
                )
                """
                if feature_table_exists
                else ""
            )
            cursor.execute(f"""
                MERGE INTO FEATURE_MATCH_QUEUE target
                USING (
                    SELECT DISTINCT raw.MATCH_ID
                    FROM RAW_DELIVERIES raw
                    {missing_feature_filter}
                ) source
                ON target.MATCH_ID = source.MATCH_ID
                WHEN NOT MATCHED THEN
                    INSERT (MATCH_ID, QUEUED_AT)
                    VALUES (source.MATCH_ID, CURRENT_TIMESTAMP())
            """)
    except Exception as e:
        print(f"Snowflake connection failed: {e}")
        return

    csv_files = filter_unseen_match_files(csv_files, existing_match_ids)
    total_matches = len(csv_files)
    if not csv_files:
        print("No unseen IPL matches found; raw ingestion is already current.")
        cursor.close()
        ctx.close()
        return
    print(f"📦 Discovered {total_matches} historical match profiles to stream.")
    
    print("⏳ Parsing data files and enforcing uniform column types...")
    master_df_list = []
    
    for idx, match_file in enumerate(csv_files):
        with zip_data.open(match_file) as f:
            df_match = pd.read_csv(
                f,
                low_memory=False,
                dtype={
                    'season': str,
                    'ball': str,
                    'actual_delivery': str,
                },
            )

        df_match.columns = df_match.columns.str.strip().str.replace(' ', '_').str.upper()

        # BALL is Cricsheet's unique delivery identifier. ACTUAL_DELIVERY is
        # the legal-ball commentary number and repeats after wides/no-balls.
        # DELIVERY_SEQ preserves exact source chronology.
        df_match['DELIVERY_SEQ'] = range(len(df_match))

        # Defensive fallback: some cricsheet CSV vintages don't include a MATCH_ID column
        # (the id lives in the filename instead). Never let MATCH_ID silently end up null.
        file_match_id = os.path.splitext(os.path.basename(match_file))[0]
        if 'MATCH_ID' not in df_match.columns:
            df_match['MATCH_ID'] = file_match_id
        else:
            df_match['MATCH_ID'] = df_match['MATCH_ID'].fillna(file_match_id)

        master_df_list.append(df_match)
        
        if (idx + 1) % 200 == 0:
            print(f"|--- Compiled {idx + 1}/{total_matches} match data arrays...")

    print("🥞 Stacking data frames into a consolidated database matrix...")
    final_raw_df = prepare_raw_deliveries(master_df_list)

    print("🧹 Aligning DataFrame matrix directly to production warehouse schema...")
    print("🔌 Connecting securely to the Snowflake Cloud Data Platform...")


    print(f"⚡ Streaming {len(final_raw_df):,} total row vectors up to the warehouse platform...")
    success, nchunks, nrows, _ = write_pandas(
        conn=ctx,
        df=final_raw_df,
        table_name='RAW_DELIVERIES',
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )
    if not success:
        raise RuntimeError("Snowflake did not confirm the incremental upload")

    new_match_ids = tuple(
        sorted(final_raw_df["MATCH_ID"].dropna().astype(str).unique())
    )
    placeholders = ",".join(["%s"] * len(new_match_ids))
    cursor.execute(
        f"""
        MERGE INTO FEATURE_MATCH_QUEUE target
        USING (
            SELECT DISTINCT MATCH_ID
            FROM RAW_DELIVERIES
            WHERE MATCH_ID IN ({placeholders})
        ) source
        ON target.MATCH_ID = source.MATCH_ID
        WHEN NOT MATCHED THEN
            INSERT (MATCH_ID, QUEUED_AT)
            VALUES (source.MATCH_ID, CURRENT_TIMESTAMP())
        """,
        new_match_ids,
    )
    
    print(f"🎉 Success! Ingested {nrows:,} rows across {nchunks} streaming compute clusters directly into Snowflake.")
    
    cursor.close()
    ctx.close()
    print("🏆 Step 1 Production Cloud Migration Complete!")

if __name__ == "__main__":
    upload_raw_data_to_snowflake()
