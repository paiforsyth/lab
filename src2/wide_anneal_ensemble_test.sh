#!/bin/bash
python . --ensemble_args_files="../argfiles/anneal_wide_c1"  --ensemble_args_files="../argfiles/anneal_wide_c2"  --ensemble_args_files="../argfiles/anneal_wide_c3" --ensemble_args_files="../argfiles/anneal_wide_c4" --ensemble_args_files="../argfiles/anneal_wide_c5" --resume_mode=ensemble   --ensemble_models_files=16_February_2018_Friday_02_15_42mnist-checkpoint-run_checkpoint_1      --ensemble_models_files=16_February_2018_Friday_02_15_42mnist-checkpoint-run_checkpoint_2

