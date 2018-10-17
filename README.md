# A2Net implementation forked by CNNGeometric

![](http://cvlab.postech.ac.kr/research/A2Net/images/a2net.png)

This is the implementation of the paper:

 P. H. Seo, J. Lee, D. Jung, B. Han and M. Cho.  Attentive Semantic Alignment with Offset-Aware Correlation Kernels. ECCV 2018 [[website](http://cvlab.postech.ac.kr/research/A2Net/)][[arXiv](https://arxiv.org/abs/1808.02128)]

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
  - Results on PF-PASCAL(by bounding box): `PCK affine (vgg) : 0.456`, `PCK affine+tps (vgg) : 0.540`, `PCK affine+tps (ResNet101): 0.589`
  - Results on PF-PASCAL(by image size): `PCK affine (vgg) : 0.587`, `PCK affine+tps (vgg) : 0.650`, `PCK affine+tps (ResNet101): 0.708`

## NOTICE!! ###
Note that there has been some inconsistency across related papers in determining the error tolerance threshold for PCK: some paper ( DeepFlow, GMK, SIFTFlow, DSP, ProposalFlow ) determine the threshold based on the object bounding box size whereas some others use the entire image size (UCN, FCSS, SCNet). 

Unfortunately, this issue confuses the comparisons across the methods. In producing the PF-PASCAL benchmark comparison in our paper, we’ve used the codes from the previous methods and in doing so we made some mistake in some of evaluation due to the issue. 

Although the overall tendencies of performances between models remain unchanged, scores of some models are overestimated. We have posted a new version of our paper in arXiv with all the correct scores measured with bounding box sizes. Please refer to this version for the correct scores: https://arxiv.org/abs/1808.02128 . We apologize for all this inconvenience.



## BibTeX ##
````
@inproceedings{paul2018attentive,
   title={Attentive Semantic Alignment with Offset-Aware Correlation Kernels},
   author={Paul Hongsuck Seo and Jongmin Lee and Deunsol Jung and Bohyung Han and Minsu Cho},
   booktitle={ECCV}
   year={2018}
}
````
