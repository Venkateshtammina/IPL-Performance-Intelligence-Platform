# Incremental IPL Data Updates

Opening the Streamlit website does not run the data pipeline. When Cricsheet
publishes new IPL matches, run:

```powershell
python src/ingestion.py
python src/features.py
python src/player_registry.py
python src/models.py
```

`ingestion.py` skips match IDs already present in `RAW_DELIVERIES`, appends only
unseen matches, and adds those match IDs to `FEATURE_MATCH_QUEUE`.

`features.py` materializes only queued matches, replaces only those matches in
`ANALYTICAL_MATCHUP_FEATURES`, and clears the queue after a successful
transaction. Historical window calculations still read the complete raw
history.

Model calibration is intentionally retrained on the complete feature history
after new matches arrive. Scikit-Learn encoders and calibrated probability
models are not updated safely one match at a time.

For corrected historical files or a complete rebuild:

```powershell
$env:FULL_REFRESH = "true"
python src/ingestion.py
python src/features.py
python src/player_registry.py
python src/models.py
Remove-Item Env:FULL_REFRESH
```
