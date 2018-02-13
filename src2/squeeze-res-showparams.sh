#!/usr/bin/env bash 
python . --save_prefix="squeeze-resish" --dataset_for_classification=cifar_challenge  --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_conv1_stride=1  --squeezenet_in_channels=3 --squeezenet_prop3=0.5 --squeezenet_resfire --squeezenet_dropout_rate=0 --squeezenet_base=16  --squeezenet_incr=16 --squeezenet_freq=6 --squeezenet_sr=0.5 --fire_skip_mode=simple --squeezenet_num_fires=18 --squeezenet_pool_interval=6 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_pooling_count_offset=1 --batch_size=64  --num_epochs=100   --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.1  --lr_scheduler=plateau --plateau_lr_scheduler_patience=5  --grad_norm_clip=50 
