#!/usr/bin/env bash 
python . --save_prefix="zounds_squeeze-mnist" --dataset_for_classification=mnist  --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_dropout_rate=0.5 --num_epochs=10 --squeezenet_prop3=0.1  --param_difs --optimizer=rmsprop --init_lr=0.0001  --lr_scheduler=exponential  --lr_gamma=0.9 --grad_norm_clip=50 --fire_skip_mode=simple
