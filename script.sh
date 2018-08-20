# VGG16 + affine transformation
# python train.py --lr 0.0002 --num-epochs 50 --batch-size 32 --geometric-model affine --feature-extraction-cnn vgg --random-sample True

# VGG16 + tps transformation
# python train.py --lr 0.0002 --num-epochs 50 --batch-size 32 --geometric-model affine --feature-extraction-cnn vgg --random-sample True

# Resnet101 + affine transformation
python train.py --lr 0.0002 --num-epochs 50 --batch-size 32 --geometric-model affine --feature-extraction-cnn resnet101 --random-sample True

# Resnet101 + tps transformation
python train.py --lr 0.0002 --num-epochs 50 --batch-size 32 --geometric-model tps --feature-extraction-cnn resnet101 --random-sample True
