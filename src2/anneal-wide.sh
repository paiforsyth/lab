#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j.out
module purge
module load miniconda3
source activate pytorch
python . --save_prefix=anneal-wide --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3 --squeezenet_prop3=0.5 --squeezenet_mode=resfire --squeezenet_dropout_rate=0 --squeezenet_base=128 --squeezenet_incr=128 --squeezenet_freq=10 --squeezenet_sr=0.2 --fire_skip_mode=simple --squeezenet_num_fires=30 --squeezenet_pool_interval=10 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=96 --squeezenet_pooling_count_offset=1 --squeezenet_use_excitation --batch_size=128 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=6 --cuda



