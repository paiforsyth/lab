#!/bin/bash
#SBATCH --gres=gpu:lgpu:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j.out
module purge
module load miniconda3
source activate pytorch
python . --save_prefix=rapid_se_stochastic_pyramidnet_attempt_deeper --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_next_fire_final_bn  --squeezenet_pool_interval_mode=add  --squeezenet_base=64 --squeezenet_freq=1 --squeezenet_incr=5 --squeezenet_sr=0.25 --fire_skip_mode=zero_pad --squeezenet_num_fires=115 --squeezenet_pool_interval=45 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16 --squeezenet_pooling_count_offset=0 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=add  --batch_size=64 --num_epochs=175 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.15 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=5   --epoch_anneal_save_last --squeezenet_num_layer_chunks=4 --squeezenet_use_non_default_layer_splits --squeezenet_layer_splits 0 60 80 105  --cifar_random_erase  --squeezenet_next_fire_stochastic_depth --squeezenet_use_excitation --squeezenet_chunk_across_devices --squeezenet_layer_chunk_devices 0 1 2 3 --cuda --resume_mode standard --res_file 28_February_2018_Wednesday_19_56_57se_stochastic_pyramidnet_attempt_deeper_endofcycle_checkpoint_0 --save_every_epoch 

