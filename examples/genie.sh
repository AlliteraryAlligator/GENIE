#!/bin/bash

source venv
cd ~/GENIE/
mkdir -p logs

# Generate Responses
python -m scripts.generate_responses --prompts_path data/prompts.json \
    --config_path configs/population_models.yaml \
    --data_path data/population_responses > logs/population_responses.log 2>&1 &
PID1=$!

python -m scripts.generate_responses --prompts_path data/prompts.json \
    --config_path configs/target_models.yaml \
    --data_path data/target_responses > logs/target_responses.log 2>&1 &
PID2=$!

wait $PID1 $PID2

deactivate

source venv2

# Process Responses
python -m scripts.process_responses --input data/population_responses \
    --output data/population_responses_processed.json > logs/pop_processing.log 2>&1 &
PID1=$!

python -m scripts.process_responses --input data/target_responses \
    --output data/target_responses_processed.json > logs/target_processing.log 2>&1 &
PID2=$!

wait $PID1 $PID2

deactivate

# Find Answers in Responses
python -m scripts.answer --documents data/population_responses_processed.json \
    --questions data/questions.json \
    --output data/population_answers.json > logs/pop_answering.log 2>&1 &
PID1=$!

python -m scripts.answer --documents data/target_responses_processed.json \
    --questions data/questions.json \
    --output data/target_answers.json > logs/target_answering.log 2>&1 &
PID2=$!

wait $PID1 $PID2

# Compute similarity & GENIE
python -m scripts.genie --population_responses data/population_responses_processed.json \
    --population_answers data/population_answers.json \
    --sampled_population_responses data/sampled_population_responses.json \
    --sampled_population_answers data/sampled_population_answers.json \
    --target_answers data/target_answers.json \
    --sim_pairs data/sim_pairs.json \
    --similarity data/sim_pairs.json \
    --genie_output data/genie.json \
    --q_genie_output data/q_genie.json 

deactivate
