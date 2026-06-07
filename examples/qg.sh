#!/bin/bash

source venv
cd ~/GENIE/
python -m scripts.qg --prompts configs/prompts.json \
    --qg_model gpt-4.1-mini \
    --filtering_model gpt-4.1-mini \
    --mapping_model gpt-4.1-mini \
    --qg_output data/questions.json
deactivate