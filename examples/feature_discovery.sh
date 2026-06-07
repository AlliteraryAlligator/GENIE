#!/bin/bash

source venv
cd ~/GENIE/
python -m src.feature_discovery.discover_features --model gpt-4.1-mini \
    --emb_model text-embedding-3-small \
    --num_samples 10 \
    --algorithm hdbscan \
    --task 'creative writing' \
    --out_file data/features/creative_writing_features.json \
    --mode feature

deactivate