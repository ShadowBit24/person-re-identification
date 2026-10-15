import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import numpy as np
import os

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')  # Changed for fan_in as default
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, nonlinearity='linear')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.add_block(x)
        x = self.classifier(x)
        return x



# Function to load histogram features
def load_histogram_features(image_path):
    histogram_file = os.path.splitext(image_path)[0] + '_histogram.npy'
    if os.path.exists(histogram_file):
        histogram = np.load(histogram_file)
    else:
        histogram = np.zeros((256,), dtype=np.float32)  # Default to zero histogram if file not found
    return histogram

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 256, class_num)  # 2048 for ResNet50, 256 for histogram

    def forward(self, x, histogram):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        # Concatenate image features with histogram features
        x = torch.cat((x, histogram), dim=1)
        x = self.classifier(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024 + 256, class_num)  # 1024 for DenseNet121, 256 for histogram

    def forward(self, x, histogram):
        x = self.model.features(x)
        x = torch.squeeze(x)
        # Concatenate image features with histogram features
        x = torch.cat((x, histogram), dim=1)
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048 + 1024 + 256, class_num)  # 2048 + 1024 for ResNet50, 256 for histogram

    def forward(self, x, histogram):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        # Concatenate image features with histogram features
        x = torch.cat((x, histogram), dim=1)
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, hist_feature_dim=768):  # Adjusted for flattened RGB histograms
        super(PCB, self).__init__()
        self.part = 6  # Number of parts the image is divided into
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # Adjusting the model to not downsample in the final layer
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

        self.classifiers = nn.ModuleList()
        for _ in range(self.part):
            self.classifiers.append(ClassBlock(2048 + hist_feature_dim, class_num, True, False, 256))

    def forward(self, x, histogram):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        part_outputs = []
        histogram = histogram.view(histogram.size(0), -1)  # Flatten the histogram for each batch
        
        for i, classifier in enumerate(self.classifiers):
            part_feature = x[:, :, i, :].view(x.size(0), -1)
            combined_feature = torch.cat((part_feature, histogram), dim=1)
            part_output = classifier(combined_feature)
            part_outputs.append(part_output)

        # Assuming you want to return the outputs of all parts
        # This will return a list of tensors, one for each part
        return part_outputs



class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y



# debug model structure
#net = ft_net(751)
net = ft_net_dense(751)
#print(net)
input = Variable(torch.FloatTensor(8, 3, 224, 224))
histogram = Variable(torch.FloatTensor(8, 256))  # Dummy histogram tensor, replace with actual histogram data
output = net(input, histogram)
print('net output size:')
print(output.shape)
