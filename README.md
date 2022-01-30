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

[Training module] The training module is controlled by the following arguments:
* --sparse - Enable sparse mode (remove this if want to train dense model)
* --fix - Fix the sparse pattern during training (remove this if want to with dynamic sparse training)
* --sparse-init - Type of sparse initialization. Choose from: uniform, uniform_plus, ERK, ERK_plus, ER, snip (snip ratio), GraSP (GraSP ratio)
* --model (str) - cifar_resnet_A_B, where A is the depths and B is the width, e.g., cifar_resnet_20_32
* --density (float) - density level (default 0.05)

### CIFAR-10/100 Experiments
To train ResNet with various depths on CIFAR10/100:

```bash
for model in cifar_resnet_20 cifar_resnet_32 cifar_resnet_44 cifar_resnet_56 cifar_resnet_110 
do
    python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model $model --data cifar10 --epoch 160
done
```

To train ResNet with various depths on CIFAR10/100:

```bash
for model in cifar_resnet_20_8 cifar_resnet_20_16 cifar_resnet_20_24 
do
    python main.py --sparse --seed 17 --sparse_init ERK --fix --lr 0.1 --density 0.05 --model $model --data cifar10 --epoch 160
done
```

### ImageNet Experiments

To train WideResNet50_2 on ImageNet with ERK_plus:

```bash
cd ImageNet
python $1multiproc.py --nproc_per_node 4 $1main.py --sparse_init ERK_plus --fc_density 1.0 --fix --fp16 --master_port 5556 -j 10 -p 500 --arch WideResNet50_2 -c fanin --label-smoothing 0.1 -b 192 --lr 0.4 --warmup 5 --epochs 100 --density 0.2 --static-loss-scale 256 $2 ../../../../../../data1/datasets/imagenet2012/ --save save/
```

# Citation
if you find this repo is helpful, please cite

```bash
@inproceedings{
liu2022the,
title={The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training},
author={Shiwei Liu and Tianlong Chen and Xiaohan Chen and Li Shen and Decebal Constantin Mocanu and Zhangyang Wang and Mykola Pechenizkiy},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=VBZJ_3tz-t}
}
```

