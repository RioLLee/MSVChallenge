#! /usr/bin/python
# -*- encoding: utf-8 -*-
import pdb
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import sys
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8,att_type="SE"):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNetSE(nn.Module):
    def __init__(self, layers, num_filters, encoder_type='ASP',n_mels=80, attention_channels=256,lin_neurons=512):
        super(ResNetSE, self).__init__()
        print('Embedding size is %d, encoder %s.'%(lin_neurons, encoder_type))
        self.inplanes   = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels     = n_mels
        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self._make_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))
        outmap_size = 10
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError('Undefined encoder')
        #embedding layer 
        self.fc = nn.Linear(out_dim, lin_neurons)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
    def forward(self, x,label=None):
        x = x.transpose(-1,-2).contiguous().unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size()[0],-1,x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,sg),1)

        x = x.view(x.size()[0], -1)
        #embedding layer
        x = self.fc(x)
        return x.unsqueeze(1) 
#(self, block, layers, num_filters, encoder_type='ASP', attention_channels=128,lin_neurons=512)

def MainModel(layers=[3,4,6,3], num_filters=[32, 64, 128, 256], encoder_type='ASP', attention_channels=128,lin_neurons=512):
    #num_filters = [32, 64, 128, 256]
    #num_filters = [64, 128, 256, 512]
    model = ResNetSE(layers, num_filters, encoder_type='ASP',n_mels=80, attention_channels=128,lin_neurons=512)
    return model





if __name__ == "__main__":
    model = MainModel()
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))



# class Classifier(torch.nn.Module):
#     """This class implements the cosine similarity on the top of features.

#     Arguments
#     ---------
#     device : str
#         Device used, e.g., "cpu" or "cuda".
#     lin_blocks : int
#         Number of linear layers.
#     lin_neurons : int
#         Number of neurons in linear layers.
#     out_neurons : int
#         Number of classes.

#     Example
#     -------
#     >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
#     >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
#     >>> outupts = outputs.unsqueeze(1)
#     >>> cos = classify(outputs)
#     >>> (cos < -1.0).long().sum()
#     tensor(0)
#     >>> (cos > 1.0).long().sum()
#     tensor(0)
#     """

#     def __init__(
#         self,
#         input_size,
#         device="cpu",
#         lin_blocks=0,
#         lin_neurons=192,
#         out_neurons=1211,
#     ):

#         super().__init__()
#         self.blocks = nn.ModuleList()

#         for block_index in range(lin_blocks):
#             self.blocks.extend(
#                 [
#                     _BatchNorm1d(input_size),
#                     Linear(input_size=input_size, n_neurons=lin_neurons),
#                 ]
#             )
#             input_size = lin_neurons
#         # Final Layer
#         self.weight = nn.Parameter(
#             torch.FloatTensor(out_neurons, input_size, device=device)
#         )
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, x):
#         """Returns the output probabilities over speakers.

#         Arguments
#         ---------
#         x : torch.Tensor
#             Torch tensor.
#         """
#         for layer in self.blocks:
#             x = layer(x)

#         # Need to be normalized 
#         #[B,1,192] ---> [b,192] []
#         x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
#         #[B,1,1221]
#         return x.unsqueeze(1)
