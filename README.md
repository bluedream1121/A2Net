# AttentiveOACK implementation forked by CNNGeometric

![](http://cvlab.postech.ac.kr/research/a2net/images/a2net.png)

This is the implementation of the paper:

 P. H. Seo, J. Lee, D. Jung, B. Han and M. Cho.  Attentive Semantic Alignment with Offset-Aware Correlation Kernels. ECCV 2017 [[website](http://cvlab.postech.ac.kr/research/a2net/)][[arXiv](https://arxiv.org/abs/1808.02128)]

using PyTorch.

## Dependencies ###
  - Python 3
  - pytorch > 0.2.0, torchvision
  - numpy, skimage (included in conda)

## Getting started ###
  - demo.py demonstrates the results on the ProposalFlow dataset
  - train.py is the main training script
  - eval_pf.py evaluates on the ProposalFlow dataset

## Trained models ###

#### A2Net with VGG-16
  - [[Affine]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_affine_grid_loss.pth.tar), [[TPS]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_tps_grid_loss.pth.tar)
  - Results on PF: `PCK affine: 0.478`, `PCK tps: 0.428`, `PCK affine+tps: 0.568`

#### A2Net with ResNet-101
  - [[Affine]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_affine_grid_loss_resnet_random.pth.tar), [[TPS]](http://www.di.ens.fr/willow/research/cnngeometric/trained_models/pytorch/best_pascal_checkpoint_adam_tps_grid_loss_resnet_random.pth.tar)
  - Results on PF: `PCK affine: 0.559`, `PCK tps: 0.582`, `PCK affine+tps: 0.676`

## BibTeX ##
````
@inproceedings{paul2018attentive,
   title={Attentive Semantic Alignment with Offset-Aware Correlation Kernels},
   author={Paul Hongsuck Seo and Jongmin Lee and Deunsol Jung and Bohyung Han and Minsu Cho},
   booktitle={ECCV}
   year={2018}
}
````
