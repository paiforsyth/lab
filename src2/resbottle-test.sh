#!/usr/bin/env bash 
python . --save_prefix=res-with-bottlenecking --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3 --squeezenet_prop3=0.5 --squeezenet_mode=resfire --squeezenet_dropout_rate=0 --squeezenet_base=64 --squeezenet_incr=64 --squeezenet_freq=10 --squeezenet_sr=0.125 --fire_skip_mode=simple --squeezenet_num_fires=30 --squeezenet_pool_interval=10 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=32 --squeezenet_pooling_count_offset=1 --batch_size=128 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.1 --lr_scheduler=plateau --plateau_lr_scheduler_patience=50  --res_file="../saved_models/14_February_2018_Wednesday_15_30_05res-with-bottlenecking_best_model" --test_report_filename="../reports/resbottlereport.csv" --mode=test --resume 
