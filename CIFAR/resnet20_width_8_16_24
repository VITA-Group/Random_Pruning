#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J width_cf100_dense_81624              # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p all                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 1-12:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o width_cf100_dense_81624.out        # 打印输出的文件
source /home/TUE/20180170/miniconda3/etc/profile.d/conda.sh
source activate torch151
# choices of sparse_init are [snip, ERK, ER, uniform, uniform_plus]
sparse_init=ERK
data=cifar10

for model in cifar_resnet_20 cifar_resnet_32 cifar_resnet_44 cifar_resnet_56 cifar_resnet_110
do
    python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model $model --data cifar10 --epoch 160
done

conda deactivate torch151
