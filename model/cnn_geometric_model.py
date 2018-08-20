from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models.vgg import model_urls
from torchvision.models.resnet import model_urls as resnet_urls

class FeatureExtraction(torch.nn.Module):
    def __init__(self, use_cuda=True, feature_extraction_cnn='vgg', last_layer=''):
        super(FeatureExtraction, self).__init__()
        if feature_extraction_cnn == 'vgg':
            model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers=['conv1_1','relu1_1','conv1_2','relu1_2','pool1','conv2_1',
                         'relu2_1','conv2_2','relu2_2','pool2','conv3_1','relu3_1',
                         'conv3_2','relu3_2','conv3_3','relu3_3','pool3','conv4_1',
                         'relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','pool4',
                         'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','pool5']
            if last_layer=='':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx+1])
        if feature_extraction_cnn == 'resnet101':
            resnet_urls['resnet101'] = resnet_urls['resnet101'].replace('https://', 'http://')
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])
        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model.cuda()

    def forward(self, image_batch):
        return self.model(image_batch)

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
#        print(feature.size())
#        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor

class CorrelationAlign(nn.Module):
    def __init__(self):
        super(CorrelationAlign, self).__init__()

    def forward(self, correlation_tensor):
        # correlation_tensor: b, h*w, h, w
        b, _, h, w = correlation_tensor.size()

        resized_tensor = correlation_tensor.contiguous().view(b, h, w, h, w)
        enlarged_h = (2*h-1)
        enlarged_w = (2*w-1)
        correlation_tensor_ = Variable(torch.zeros(b, enlarged_h, enlarged_w, h, w)).cuda()
        for i in range(h):
            for j in range(w):
                correlation_tensor_[:, enlarged_h-h-i:enlarged_h-i, enlarged_w-w-j:enlarged_w-j, i, j] = resized_tensor[:, :, :, i, j]

        return correlation_tensor_.view(b, -1, h, w)

class Attention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32):
        super(Attention, self).__init__()
        self.att = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)
        )
        self.att.cuda()

    def forward(self, score_feat, merge_feat):
        b, c, h, w = score_feat.size()
        att = self.att(score_feat)
        att = F.softmax(att.view(-1, h*w), 1).view(-1, 1, h, w).expand_as(merge_feat)
        output = att * merge_feat
        output = output.sum(3).sum(2)
        return output


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.align = CorrelationAlign()
        self.preconv = nn.Sequential(
            nn.Conv2d(841, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128+5, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(128, output_dim)
        self.att = Attention(128, 64)
        self.weight = Parameter(torch.ones(1, 5, 9, 9))
        self.weight.data.uniform_(-1, 1)
        if use_cuda:
            self.preconv.cuda()
            self.conv.cuda()
            self.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.align(x)
        x = self.preconv(x)
        x_ = x
        x = torch.cat((self.weight.expand(x.size(0), 5, 9, 9), x), 1)
        x = self.conv(x)
        x = self.att(x_, self.proj(x))
        x = self.linear(x)
        return x


class CNNGeometric(nn.Module):
    def __init__(self, geometric_model='affine', normalize_features=True, normalize_matches=True, batch_normalization=True, use_cuda=True, feature_extraction_cnn='vgg'):
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.FeatureExtraction = FeatureExtraction(use_cuda=self.use_cuda, feature_extraction_cnn=feature_extraction_cnn)
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        if geometric_model=='affine':
            output_dim = 6
        elif geometric_model=='tps':
            output_dim = 18
        self.FeatureRegression = FeatureRegression(output_dim,use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, tnf_batch):
        # do feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        # normalize
        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        # do feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
#        correlation = self.FeatureL2Norm(correlation)
        # do regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)

        return theta
