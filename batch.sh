#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=10000
#SBATCH --output=results/segment_7_%04a_stdout.txt
#SBATCH --error=results/segment_7_%04a_stderr.txt
#SBATCH --time=24:00:00
#SBATCH --job-name=segment_7_exp
#SBATCH --mail-user=rachel.l.bennett-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504303/aml/CS5043_Assignment8
#SBATCH --array=0-4

. /home/fagg/tf_setup.sh
conda activate tf

test 
python hw8_base.py @parameters.txt --label "round3" --model_type 0 --exp_index $SLURM_ARRAY_TASK_ID

