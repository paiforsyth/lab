#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=3-00:00
#SBATCH --output=%N-%j.out
#module purge
#module load miniconda3
source activate pytorch
python . --save_prefix=anneal-wider-train-from-pyramid --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=1 --squeezenet_freq=3 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=9 --squeezenet_pool_interval=3 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=63 --squeezenet_pooling_count_offset=0 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply  --squeezenet_use_excitation --batch_size=256  --num_epochs=1000 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_init_period=2 --epoch_anneal_mult_factor=2 --epoch_anneal_save_last --save_every_epoch --cifar_random_erase --born_again_enable --born_again_model_file --cuda ../saved_models/28_February_2018_Wednesday_19_56_57se_stochastic_pyramidnet_attempt_deeper_endofcycle_checkpoint_0  --born_again_args_file pyramidnet-attempt_argsfile     



