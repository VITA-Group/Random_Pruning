# [ICLR 2022] The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training

<img src="https://github.com/Shiweiliuiiiiiii/Random_Pruning/blob/main/depth_cf10-1.png" width="800" height="400">


**The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training**<br>
Shiwei Liu, Tianlong Chen, Xiaohan Chen, Li Shen, Decebal Constantin Mocanu, Zhangyang Wang, Mykola Pechenizkiy

https://openreview.net/forum?id=VBZJ_3tz-t

Abstract: *Random pruning is arguably the most naive way to attain sparsity in neural networks, but has been deemed uncompetitive by either post-training pruning or sparse training. In this paper, we focus on sparse training and highlight a perhaps counter-intuitive finding, that random pruning at initialization (PaI) can be quite powerful for the sparse training of modern neural networks. Without any delicate pruning criteria or carefully pursued sparsity structures, we empirically demonstrate that sparsely training a randomly pruned network from scratch can match the performance of its dense equivalent. There are two key factors that contribute to this revival: (i) the network sizes matter: as the original dense networks grow wider and deeper, the performance of training a randomly pruned sparse network will quickly grow to matching that of its dense equivalent, even at high sparsity ratios; (ii) appropriate layer-wise sparsity ratios can be pre-chosen for sparse training, which shows to be another important performance booster. Simple as it looks, a randomly pruned subnetwork of Wide ResNet-50 can be sparsely trained to match the accuracy of a dense Wide ResNet-50, on ImageNet. We also observed such randomly pruned networks outperform dense counterparts in other favorable aspects, such as out-of-distribution detection, uncertainty estimation, and adversarial robustness. Overall, our results strongly suggest there is larger-than-expected room for sparse training at scale, and the benefits of sparsity might be more universal beyond carefully designed pruning.*

This code base is created by Shiwei Liu s.liu3@tue.nl during his Ph.D. at Eindhoven University of Technology.

## Requirements
Python 3.6, PyTorch v1.5.1, and CUDA v10.2.

## How to Run Experiments

### CIFAR-10/100 Experiments
To train Wide ResNet28-10 on CIFAR10/100 with DST ensemble at sparsity 0.8:

```bash
python main_DST.py --sparse --model wrn-28-10 --data cifar10 --seed 17 --sparse-init ERK \
--update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.8 \
--growth gradient --death magnitude --redistribution none --epochs 250 --density 0.2

```

To train Wide ResNet28-10 on CIFAR10/100 with EDST ensemble at sparsity 0.8:

```bash
python3 main_EDST.py --sparse --model wrn-28-10 --data cifar10 --nolrsche \
--decay-schedule constant --seed 17 --epochs-explo 150 --model-num 3 --sparse-init ERK \
--update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.8 \
--growth gradient --death magnitude --redistribution none --epochs 450 --density 0.2
```
[Training module] The training module is controlled by the following arguments:
* `--epochs-explo` - An integer that controls the training epochs of the exploration phase.
* `--model-num` - An integer, the number free tickets to produce.
* `--large-death-rate` - A float, the ratio of parameters to explore for each refine phase.
* `--density` - An float, the density (1-sparsity) level for each free ticket.

To train Wide ResNet28-10 on CIFAR10/100 with PF (prung and finetuning) ensemble at sparsity 0.8:

First, we need train a dense model with:

```bash
python3 main_individual.py  --model wrn-28-10 --data cifar10 --decay-schedule cosine --seed 18 \
--sparse-init ERK --update-frequency 1000 --batch-size 128 --death-rate 0.5 --large-death-rate 0.5 \
--growth gradient --death magnitude --redistribution none --epochs 250 --density 0.2
```

Then, perform pruning and finetuning with:

```bash
pretrain='results/wrn-28-10/cifar10/individual/dense/18.pt'
python3 main_PF.py --sparse --model wrn-28-10 --resume --pretrain $pretrain --lr 0.001 \
--fix --data cifar10 --nolrsche --decay-schedule constant --seed 18 
--epochs-fs 150 --model-num 3 --sparse-init pruning --update-frequency 1000 --batch-size 128 \
--death-rate 0.5 --large-death-rate 0.8 --growth gradient --death magnitude \
--redistribution none --epochs $epoch --density 0.2
```

After finish the training of various ensemble methods, run the following commands for test ensemble:

```bash
resume=results/wrn-28-10/cifar10/density_0.2/EDST/M=3/
python ensemble_freetickets.py --mode predict --resume $resume --dataset cifar10 --model wrn-28-10 \
--seed 18 --test-batch-size 128
```
* `--resume` - An folder path that contains the all the free tickets obtained during training.
* `--mode` - An str that control the evaluation mode, including: predict, disagreement, calibration, KD, and tsne.
* 
### ImageNet Experiments

```bash
cd ImageNet
python $1multiproc.py --nproc_per_node 2 $1main.py --sparse_init ERK --multiplier 1 --growth gradient --seed 17 --master_port 4545 -j5 -p 500 --arch resnet50 -c fanin --update_frequency 4000 --label-smoothing 0.1 -b 64 --lr 0.1 --warmup 5 --epochs 310 --density 0.2 $2 ../data/
```

# Citation
if you find this repo is helpful, please cite

```bash
@inproceedings{
liu2022deep,
title={Deep Ensembling with No Overhead for either Training or Testing: The All-Round Blessings of Dynamic Sparsity},
author={Shiwei Liu and Tianlong Chen and Zahra Atashgahi and Xiaohan Chen and Ghada Sokar and Elena Mocanu and Mykola Pechenizkiy and Zhangyang Wang and Decebal Constantin Mocanu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=RLtqs6pzj1-}
}
```

