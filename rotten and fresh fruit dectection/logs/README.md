# Scan Logs

This directory stores scan event logs recorded during live scanner execution (`python app.py scan`).

## Files

- `scans.csv`: CSV log containing timestamped scan events.

### CSV Format
```csv
time,batch,result,fresh_score,rotten_score,threshold,inference_ms
2026-09-26T10:30:00+05:30,batch_a1b2c3d4,FRESH,0.9123,0.0877,0.80,37.4
```

> [!NOTE]
> Scan logs preserve historical detection records. Resetting a batch in the scanner (by pressing `B`) does not wipe existing rows in `scans.csv`.
