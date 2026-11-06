#!/usr/bin/env bash
echo "Starting Nepal Temperature Anomaly & Global Warming Dashboard (Tripti Malla 260197)..."
cd "$(dirname "$0")"
if command -v streamlit >/dev/null 2>&1; then
    streamlit run dashboard.py
elif [ -f "../.venv/bin/streamlit" ]; then
    ../.venv/bin/streamlit run dashboard.py
else
    echo "Streamlit not found. You can run: pip install -r requirements.txt"
    echo "Alternatively, open dashboard.html directly in your browser."
fi
