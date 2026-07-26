import os
import snowflake.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def run_snowflake_feature_engineering():
    print("🚀 Re-engineering Cloud Feature Layer with Target Labels...")
    
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS FEATURE_MATCH_QUEUE (
                MATCH_ID VARCHAR,
                QUEUED_AT TIMESTAMP_NTZ
            )
        """)
        cursor.execute("SELECT COUNT(*) FROM FEATURE_MATCH_QUEUE")
        queued_matches = int(cursor.fetchone()[0])
        if queued_matches == 0:
            print("No queued matches found; analytical features are current.")
            cursor.close()
            ctx.close()
            return
        print("✅ Connected to Snowflake Virtual compute clusters.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    feature_generation_sql = """
    CREATE OR REPLACE TEMPORARY TABLE INCREMENTAL_ANALYTICAL_FEATURES AS
    WITH deduplicated_deliveries AS (
        SELECT *
        FROM RAW_DELIVERIES
        WHERE INNINGS IN (1, 2)
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY MATCH_ID, INNINGS, BALL
            ORDER BY DELIVERY_SEQ
        ) = 1
    ),
    match_innings_base AS (
        SELECT
            MATCH_ID, SEASON, START_DATE, VENUE, INNINGS, BALL, DELIVERY_SEQ, STRIKER, NON_STRIKER, BOWLER,
            RUNS_OFF_BAT, EXTRAS, WIDES, NOBALLS,
            (RUNS_OFF_BAT + EXTRAS) as TOTAL_BALL_RUNS,
            CASE
                WHEN COALESCE(WIDES, 0) = 0 AND COALESCE(NOBALLS, 0) = 0 THEN 1
                ELSE 0
            END as IS_LEGAL_BALL,
            -- Team wicket state includes every real dismissal.
            CASE WHEN WICKET_TYPE IS NOT NULL AND WICKET_TYPE NOT IN ('retired hurt') THEN 1 ELSE 0 END as IS_WICKET,
            -- Only dismissals credited to the bowler are used by the matchup model.
            CASE
                WHEN PLAYER_DISMISSED = STRIKER
                     AND LOWER(WICKET_TYPE) IN (
                         'bowled', 'caught', 'caught and bowled', 'lbw',
                         'stumped', 'hit wicket'
                     ) THEN 2
                WHEN RUNS_OFF_BAT >= 4 THEN 1
                ELSE 0
            END as EVENT_TYPE
        FROM deduplicated_deliveries
    ),
    match_outcomes AS (
        SELECT 
            MATCH_ID,
            SUM(CASE WHEN INNINGS = 1 THEN TOTAL_BALL_RUNS ELSE 0 END) as T1_TOTAL,
            SUM(CASE WHEN INNINGS = 2 THEN TOTAL_BALL_RUNS ELSE 0 END) as T2_TOTAL
        FROM match_innings_base
        GROUP BY MATCH_ID
    ),
    running_metrics AS (
        SELECT 
            b.MATCH_ID, b.SEASON, b.START_DATE, b.VENUE, b.INNINGS, b.BALL, b.STRIKER, b.NON_STRIKER, b.BOWLER, b.RUNS_OFF_BAT,
            b.IS_WICKET, b.EVENT_TYPE,
            -- DELIVERY_SEQ preserves source chronology. BALL remains a string
            -- identifier so values such as 23.1 and 23.10 stay distinct.
            SUM(b.TOTAL_BALL_RUNS) OVER (
                PARTITION BY b.MATCH_ID, b.INNINGS ORDER BY b.DELIVERY_SEQ
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as CURRENT_TEAM_SCORE,
            SUM(b.IS_WICKET) OVER (
                PARTITION BY b.MATCH_ID, b.INNINGS ORDER BY b.DELIVERY_SEQ
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) as CURRENT_WICKETS_LOST,
            120 - COALESCE(
                SUM(b.IS_LEGAL_BALL) OVER (
                    PARTITION BY b.MATCH_ID, b.INNINGS ORDER BY b.DELIVERY_SEQ
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ),
                0
            ) as BALLS_REMAINING,
            AVG(IFF(b.EVENT_TYPE = 1, 1.0, 0.0)) OVER (
                PARTITION BY b.STRIKER
                ORDER BY TRY_TO_DATE(b.START_DATE), b.MATCH_ID, b.INNINGS,
                         b.BALL, b.DELIVERY_SEQ
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            ) as STRIKER_RECENT_BOUNDARY_RATE,
            AVG(IFF(b.EVENT_TYPE = 1, 1.0, 0.0)) OVER (
                PARTITION BY b.BOWLER
                ORDER BY TRY_TO_DATE(b.START_DATE), b.MATCH_ID, b.INNINGS,
                         b.BALL, b.DELIVERY_SEQ
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            ) as BOWLER_RECENT_BOUNDARY_RATE,
            AVG(IFF(b.EVENT_TYPE = 2, 1.0, 0.0)) OVER (
                PARTITION BY b.BOWLER
                ORDER BY TRY_TO_DATE(b.START_DATE), b.MATCH_ID, b.INNINGS,
                         b.BALL, b.DELIVERY_SEQ
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            ) as BOWLER_RECENT_DISMISSAL_RATE,
            AVG(b.RUNS_OFF_BAT) OVER (
                PARTITION BY b.STRIKER
                ORDER BY TRY_TO_DATE(b.START_DATE), b.MATCH_ID, b.INNINGS,
                         b.BALL, b.DELIVERY_SEQ
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            ) as STRIKER_RECENT_RUN_RATE,
            AVG(b.RUNS_OFF_BAT) OVER (
                PARTITION BY b.BOWLER
                ORDER BY TRY_TO_DATE(b.START_DATE), b.MATCH_ID, b.INNINGS,
                         b.BALL, b.DELIVERY_SEQ
                ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
            ) as BOWLER_RECENT_RUN_RATE,
            AVG(b.RUNS_OFF_BAT) OVER (
                PARTITION BY b.VENUE
                ORDER BY TRY_TO_DATE(b.START_DATE), b.MATCH_ID, b.INNINGS,
                         b.BALL, b.DELIVERY_SEQ
                ROWS BETWEEN 600 PRECEDING AND 1 PRECEDING
            ) as VENUE_RECENT_RUN_RATE,
            o.T1_TOTAL as INNINGS_1_TARGET
        FROM match_innings_base b
        JOIN match_outcomes o ON b.MATCH_ID = o.MATCH_ID
    )
    SELECT 
        r.MATCH_ID, r.SEASON, r.START_DATE, r.VENUE, r.INNINGS, r.BALL, r.STRIKER, r.NON_STRIKER, r.BOWLER, r.BALLS_REMAINING,
        r.RUNS_OFF_BAT, r.EVENT_TYPE,
        COALESCE(r.CURRENT_WICKETS_LOST, 0) as CURRENT_WICKETS_LOST,
        COALESCE(r.CURRENT_TEAM_SCORE, 0) as CURRENT_TEAM_SCORE,
        COALESCE(r.STRIKER_RECENT_BOUNDARY_RATE, 0.0)
            as STRIKER_RECENT_BOUNDARY_RATE,
        COALESCE(r.BOWLER_RECENT_BOUNDARY_RATE, 0.0)
            as BOWLER_RECENT_BOUNDARY_RATE,
        COALESCE(r.BOWLER_RECENT_DISMISSAL_RATE, 0.0)
            as BOWLER_RECENT_DISMISSAL_RATE,
        COALESCE(r.STRIKER_RECENT_RUN_RATE, 0.0)
            as STRIKER_RECENT_RUN_RATE,
        COALESCE(r.BOWLER_RECENT_RUN_RATE, 0.0)
            as BOWLER_RECENT_RUN_RATE,
        COALESCE(r.VENUE_RECENT_RUN_RATE, 0.0)
            as VENUE_RECENT_RUN_RATE,
        
        CASE 
            WHEN (120 - r.BALLS_REMAINING) > 0 THEN (COALESCE(r.CURRENT_TEAM_SCORE, 0) / (120 - r.BALLS_REMAINING)) * 6 
            ELSE 0.0 
        END as CURRENT_RUN_RATE,
        
        -- Calculate structural targets and RUNS_NEEDED explicitly
        CASE 
            WHEN r.INNINGS = 2 THEN COALESCE(r.INNINGS_1_TARGET + 1, 0)
            ELSE 0 
        END as TARGET_SCORE,
        
        CASE 
            WHEN r.INNINGS = 2 THEN GREATEST(COALESCE(r.INNINGS_1_TARGET + 1, 0) - COALESCE(r.CURRENT_TEAM_SCORE, 0), 0)
            ELSE 0 
        END as RUNS_NEEDED,
        
        CASE 
            WHEN r.INNINGS = 2 AND r.BALLS_REMAINING > 0 THEN (GREATEST(COALESCE(r.INNINGS_1_TARGET + 1, 0) - COALESCE(r.CURRENT_TEAM_SCORE, 0), 0) / r.BALLS_REMAINING) * 6
            ELSE 0.0 
        END as REQUIRED_RUN_RATE,
        
        -- Materialize ultimate CHASE_WON label cleanly (1 = Batting team won, 0 = Lost/Tied).
        -- NOTE: strictly greater-than -- a tie (T2_TOTAL == T1_TOTAL) is NOT a win for the chasing side.
        CASE 
            WHEN r.INNINGS = 2 AND o.T2_TOTAL > o.T1_TOTAL THEN 1
            ELSE 0
        END as CHASE_WON
    FROM running_metrics r
    JOIN match_outcomes o ON r.MATCH_ID = o.MATCH_ID
    WHERE r.INNINGS IN (1, 2)
      AND r.MATCH_ID IN (SELECT MATCH_ID FROM FEATURE_MATCH_QUEUE);
    """

    print("⚡ Materializing structural feature targets down to Snowflake storage layers...")
    try:
        cursor.execute(feature_generation_sql)
        cursor.execute("""
            SELECT
                COUNT_IF(CURRENT_WICKETS_LOST > 10) AS INVALID_WICKET_STATES,
                COUNT_IF(BALLS_REMAINING < 0 OR BALLS_REMAINING > 120) AS INVALID_BALL_STATES
            FROM INCREMENTAL_ANALYTICAL_FEATURES
        """)
        invalid_wickets, invalid_balls = cursor.fetchone()
        if invalid_wickets or invalid_balls:
            raise ValueError(
                f"Feature integrity failed: {invalid_wickets} invalid wicket states, "
                f"{invalid_balls} invalid ball states"
            )
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ANALYTICAL_MATCHUP_FEATURES
            LIKE INCREMENTAL_ANALYTICAL_FEATURES
        """)
        cursor.execute("SELECT COUNT(*) FROM INCREMENTAL_ANALYTICAL_FEATURES")
        incremental_rows = int(cursor.fetchone()[0])
        cursor.execute("BEGIN")
        cursor.execute("""
            DELETE FROM ANALYTICAL_MATCHUP_FEATURES
            WHERE MATCH_ID IN (SELECT MATCH_ID FROM FEATURE_MATCH_QUEUE)
        """)
        cursor.execute("""
            INSERT INTO ANALYTICAL_MATCHUP_FEATURES
            SELECT * FROM INCREMENTAL_ANALYTICAL_FEATURES
        """)
        cursor.execute("DELETE FROM FEATURE_MATCH_QUEUE")
        cursor.execute("COMMIT")
        print(f"Incrementally materialized {incremental_rows:,} feature rows.")
        cursor.execute("SELECT COUNT(*) FROM ANALYTICAL_MATCHUP_FEATURES;")
        print(f"🎉 Success! Ingested {cursor.fetchone()[0]:,} labeled vectors into ANALYTICAL_MATCHUP_FEATURES.")
    except Exception as e:
        print(f"❌ Feature processing failed: {e}")
        try:
            cursor.execute("ROLLBACK")
        except Exception:
            pass
    finally:
        cursor.close()
        ctx.close()

if __name__ == "__main__":
    run_snowflake_feature_engineering()
