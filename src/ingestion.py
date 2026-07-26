import os
import urllib.request
import zipfile
import io
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

load_dotenv()

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
    total_matches = len(csv_files)
    print(f"📦 Discovered {total_matches} historical match profiles to stream.")
    
    print("⏳ Parsing data files and enforcing uniform column types...")
    master_df_list = []
    
    for idx, match_file in enumerate(csv_files):
        with zip_data.open(match_file) as f:
            df_match = pd.read_csv(f, low_memory=False, dtype={'season': str})

        df_match.columns = df_match.columns.str.strip().str.replace(' ', '_').str.upper()

        # DELIVERY_SEQ preserves the exact chronological order deliveries appear in the
        # source CSV (cricsheet rows are already in play order). This gives us a reliable
        # tiebreaker downstream for deliveries that share the same BALL value (wides/no-balls
        # don't advance the over.ball counter, so ORDER BY BALL alone is not deterministic).
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
    final_raw_df = pd.concat(master_df_list, ignore_index=True)
    final_raw_df['SEASON'] = final_raw_df['SEASON'].astype(str)
    final_raw_df.columns = final_raw_df.columns.str.upper()

    # --- SCHEMA FIX INTERCEPTOR ---
    # Define the precise list of columns our Snowflake database schema accepts
    target_columns = [
        'MATCH_ID', 'SEASON', 'START_DATE', 'VENUE', 'INNINGS', 'BALL', 'DELIVERY_SEQ',
        'ACTUAL_DELIVERY',
        'BATTING_TEAM', 'BOWLING_TEAM', 'STRIKER', 'NON_STRIKER', 'BOWLER',
        'RUNS_OFF_BAT', 'EXTRAS', 'WIDES', 'NOBALLS', 'BYES', 'LEGBYES', 'PENALTY',
        'WICKET_TYPE', 'PLAYER_DISMISSED', 'OTHER_WICKET_TYPE', 'OTHER_PLAYER_DISMISSED'
    ]
    
    # Dynamically pad any completely missing structural fields with safe default null indicators
    for col in target_columns:
        if col not in final_raw_df.columns:
            final_raw_df[col] = None
            
    # Explicitly filter the dataframe down to contain ONLY the target table configuration layout
    print("🧹 Aligning DataFrame matrix directly to production warehouse schema...")
    final_raw_df = final_raw_df[target_columns]
    final_raw_df = (
        final_raw_df.sort_values(['MATCH_ID', 'INNINGS', 'DELIVERY_SEQ'])
        .drop_duplicates(subset=['MATCH_ID', 'INNINGS', 'BALL'], keep='first')
        .reset_index(drop=True)
    )
    if final_raw_df.duplicated(['MATCH_ID', 'INNINGS', 'BALL']).any():
        raise ValueError("Duplicate deliveries remain after ingestion deduplication")

    print("🔌 Connecting securely to the Snowflake Cloud Data Platform...")
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
        print("✅ Secure cloud operational handshake complete.")
    except Exception as e:
        print(f"❌ Connection block failed. Verify authentication fields: {e}")
        return

    print("🧱 Initializing staging landing structures...")
    cursor.execute("""
    CREATE OR REPLACE TABLE RAW_DELIVERIES (
        MATCH_ID VARCHAR, SEASON VARCHAR, START_DATE VARCHAR, VENUE VARCHAR, INNINGS INT, BALL FLOAT,
        DELIVERY_SEQ INT, ACTUAL_DELIVERY FLOAT,
        BATTING_TEAM VARCHAR, BOWLING_TEAM VARCHAR, STRIKER VARCHAR, NON_STRIKER VARCHAR, BOWLER VARCHAR,
        RUNS_OFF_BAT INT, EXTRAS INT, WIDES INT, NOBALLS INT, BYES INT, LEGBYES INT, PENALTY INT,
        WICKET_TYPE VARCHAR, PLAYER_DISMISSED VARCHAR, OTHER_WICKET_TYPE VARCHAR, OTHER_PLAYER_DISMISSED VARCHAR
    );
    """)

    print(f"⚡ Streaming {len(final_raw_df):,} total row vectors up to the warehouse platform...")
    success, nchunks, nrows, _ = write_pandas(
        conn=ctx,
        df=final_raw_df,
        table_name='RAW_DELIVERIES',
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )
    
    print(f"🎉 Success! Ingested {nrows:,} rows across {nchunks} streaming compute clusters directly into Snowflake.")
    
    cursor.close()
    ctx.close()
    print("🏆 Step 1 Production Cloud Migration Complete!")

if __name__ == "__main__":
    upload_raw_data_to_snowflake()
