#!/bin/bash
#SBATCH --gres=gpu:lgpu:4       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=3-00:00
#SBATCH --output=%N-%j.out
module purge
module load miniconda3
source activate pytorch
python . --save_prefix=anneal-wider --dataset_for_classification=cifar_challenge --model_type=squeezenet --squeezenet_out_dim=100 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=8 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=1024 --squeezenet_freq=4 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=12 --squeezenet_pool_interval=5 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=1024 --squeezenet_pooling_count_offset=0 --squeezenet_max_pool_size=2 --squeezenet_pool_interval_mode=multiply  --squeezenet_use_excitation --batch_size=64 --num_epochs=300 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001 --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=3   --epoch_anneal_save_last --squeezenet_num_layer_chunks=4 --cifar_random_erase --squeezenet_chunk_across_devices --squeezenet_layer_chunk_devices 0 1 2 3  --cuda  --resume_mode=ensemble --ensemble_autogen_args --ensemble_models_files 01_March_2018_Thursday_10_04_21anneal-wider-rapid-shuffle_endofcycle_checkpoint_0 01_March_2018_Thursday_10_04_21anneal-wider-rapid-shuffle_checkpoint_0 28_February_2018_Wednesday_03_06_33anneal-wider_checkpoint_0 28_February_2018_Wednesday_03_06_33anneal-wider_endofcycle_checkpoint_0 --mode=test --test_report_filename="../reports/thurs_ensemble_report.csv"  #--mod_report #--cuda 



