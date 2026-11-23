set -eu
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"
export RF_SPECTRUM_PYTHON="${RF_SPECTRUM_PYTHON:-/home/lenovo/anaconda3/envs/rf-spectrum/bin/python}"
export RF_SIONNA_PYTHON="${RF_SIONNA_PYTHON:-/home/lenovo/anaconda3/envs/rf-sionna/bin/python}"
export PYTHONPATH="$PROJECT_DIR/spectrum-recognition/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RF_SPECTRUM_PYTHON" -B -u backend/server.py
