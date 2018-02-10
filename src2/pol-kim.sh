#!/usr/bin/env bash 
python . --save_prefix="pol_kim" --ds_path="../data/rt-polaritydata" --validation_set_size=300 --dataset_for_classification="moviepol" --model_type="kimcnn" --num_epochs=10 
