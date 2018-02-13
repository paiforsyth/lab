#!/usr/bin/env bash 
python . --save_prefix="squeezenet_test" --dataset_for_classification="mnist"  --validation_set_size=5 --model_type="squeezenet"
