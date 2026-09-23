#!/bin/bash
set -euo pipefail
export MPLBACKEND=Agg
mkdir -p tools/logs
jupyter nbconvert --execute --to notebook \
    --ExecutePreprocessor.timeout=3600 \
    --output L06_executed.ipynb --output-dir tools/logs \
    labs/L06_Aiyagari_End_To_End.ipynb
