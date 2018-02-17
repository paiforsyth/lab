#!/bin/bash
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-10:00
#SBATCH --output=%N-%j.out
#module purge
#module load miniconda3
#source activate pytorch
python . --save_prefix="mnist-checkpoint-test" --dataset_for_classification=mnist  --model_type=squeezenet --squeezenet_out_dim=10 --squeezenet_dropout_rate=0.5 --num_epochs=100 --squeezenet_prop3=0.1  --param_difs --optimizer=sgd --init_lr=0.02  --sgd_momentum=0.9 --lr_scheduler=epoch_anneal  --epoch_anneal_numcycles=5   --fire_skip_mode=simple  --resume_mode=ensemble --ensemble_autogen_args --ensemble_models_files=16_February_2018_Friday_02_15_42mnist-checkpoint-run_checkpoint_1 --ensemble_models_files=16_February_2018_Friday_02_15_42mnist-checkpoint-run_checkpoint_2 --test_report_filename="../reports/mnist_report.csv" --mode=test 
