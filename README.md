# FalconEye

FalconEye is a flight telemetry diagnostics system with:
- temporal engine state modeling,
- explainable anomaly scoring,
- streaming-friendly ingestion,
- robust CSV normalization,
- deployment-ready container infrastructure.

## Core Improvements

1. Temporal modeling
- `EngineStateModel` now keeps rolling memory and trends.
- Added degradation and sensor health dynamics over time.
- Scoring includes transients and historical z-score drift.

2. Explainability
- Each telemetry point includes:
  - `anomaly_confidence`
  - `primary_reason`
  - `reason_codes`
  - component scores (`temp_deviation_component`, `trend_component`, etc.)
- Dashboard surfaces confidence and top contributor.

3. Real-time processing
- `StreamingTelemetryProcessor` supports record-wise ingestion.
- Uses a bounded in-memory buffer for long-running sessions.
- Accepts chunked ingestion to mimic live pipelines.

4. Robustness
- Strict telemetry normalization via `normalize_telemetry_frame`.
- Numeric coercion and invalid row handling.
- Defensive checks for malformed input and invalid `dt`.

5. Deployment
- `Dockerfile` for containerized runtime.
- `docker-compose.yml` for one-command startup.
- `.dockerignore` to keep images lean.

## Project Layout

- `app.py`: Streamlit UI and visualization.
- `engine.py`: temporal simulation + explainable anomaly engine.
- `stream_processor.py`: real-time ingestion and schema-safe normalization.
- `test_engine.py`: engine behavior tests.
- `test_stream_processor.py`: ingestion/normalization tests.
- `Dockerfile`, `docker-compose.yml`: deployment assets.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
python -m unittest -v
```

## Docker Run

```bash
docker compose up --build
```

Then open `http://localhost:8501`.
