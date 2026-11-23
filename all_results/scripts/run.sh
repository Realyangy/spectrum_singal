#!/usr/bin/env bash
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SPECTRUM=/home/flyvideo/Sata2/anaconda3/envs/rf-spectrum/bin/python
PY_SIONNA=/home/flyvideo/Sata2/anaconda3/envs/rf-sionna/bin/python
export CUDA_VISIBLE_DEVICES=0
export MPLBACKEND=Agg
export YOLO_CONFIG_DIR="$DIR/yolo_config"
export PYTHONDONTWRITEBYTECODE=1
printf 'running\n' > "$DIR/status.txt"
"$PY_SPECTRUM" -u "$DIR/recognition.py" > "$DIR/recognition.log" 2>&1 &
SPECTRUM_PID=$!
if "$PY_SPECTRUM" -u "$DIR/separation.py" > "$DIR/separation.log" 2>&1; then
    if "$PY_SIONNA" -u "$DIR/separation_ber.py" >> "$DIR/separation.log" 2>&1; then SEP_STATUS=0; else SEP_STATUS=$?; fi
else
    SEP_STATUS=$?
fi
if wait "$SPECTRUM_PID"; then SPEC_STATUS=0; else SPEC_STATUS=$?; fi
if [ "$SEP_STATUS" -eq 0 ] && [ "$SPEC_STATUS" -eq 0 ]; then
    printf 'completed\n' > "$DIR/status.txt"
else
    printf 'failed: separation=%s spectrum=%s\n' "$SEP_STATUS" "$SPEC_STATUS" > "$DIR/status.txt"
fi
