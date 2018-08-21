# AttentiveOACK implementation forked by CNNGeometric

![](http://cvlab.postech.ac.kr/research/a2net/images/A2Net.png)

This is the implementation of the paper:

 P. H. Seo, J. Lee, D. Jung, B. Han and M. Cho.  Attentive Semantic Alignment with Offset-Aware Correlation Kernels. ECCV 2017 [[website](http://cvlab.postech.ac.kr/research/A2Net/)][[arXiv](https://arxiv.org/abs/1808.02128)]

using PyTorch.

## Dependencies ###
  - Python 3
  - pytorch > 0.2.0, torchvision
  - numpy, skimage (included in conda)

## Getting started ###
  - demo.py demonstrates the results on the ProposalFlow dataset
  - train.py is the main training script
  - eval_pf.py evaluates on the PF-WILLOW dataset

## Trained models ###
#### A2Net evaluated by ProposalFlow Dataset
  - VGG Models : [[Affine]](http://cvlab.postech.ac.kr/research/A2Net/data/vgg_affine_oack027.pth.tar), [[TPS]](http://cvlab.postech.ac.kr/research/A2Net/data/vgg_tps_oack027.pth.tar) , ResNet101 Models : [[Affine]](http://cvlab.postech.ac.kr/research/A2Net/data/resnet_affine_oack027.pth.tar), [[TPS]](http://cvlab.postech.ac.kr/research/A2Net/data/resnet_tps_oack027.pth.tar)
  - Results on PF-WILLOW: `PCK affine (vgg) : 0.521`, `PCK affine+tps (vgg) : 0.625`, `PCK affine+tps (ResNet101): 0.688`
  - Results on PF-PASCAL: `PCK affine (vgg) : 0.587`, `PCK affine+tps (vgg) : 0.650`, `PCK affine+tps (ResNet101): 0.708`


## BibTeX ##
````
@inproceedings{paul2018attentive,
   title={Attentive Semantic Alignment with Offset-Aware Correlation Kernels},
   author={Paul Hongsuck Seo and Jongmin Lee and Deunsol Jung and Bohyung Han and Minsu Cho},
   booktitle={ECCV}
   year={2018}
}
````
