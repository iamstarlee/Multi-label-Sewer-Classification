import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init
import torch


class Kumar2018(nn.Module):
    def __init__(self, num_classes, alpha = 1., dropout_rate = 0.5):
        super(Kumar2018, self).__init__()
        self.alpha = alpha
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding = 2),
            nn.ELU(self.alpha, inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 5, padding = 2),
            nn.ELU(self.alpha, inplace=True),
            nn.MaxPool2d(2,2))
        self.avgpool = nn.AdaptiveAvgPool2d((64, 64))
        self.classifier = nn.Sequential(
            nn.Linear(64*64*64, 1024),
            nn.ELU(self.alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ELU(self.alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Meijer2019(nn.Module):
    def __init__(self, num_classes, alpha = 1., dropout_rate = 0.5):
        super(Meijer2019, self).__init__()
        self.alpha = alpha
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding = 2),
            nn.ELU(self.alpha, inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, 5, padding = 2),
            nn.ELU(self.alpha, inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, 5, padding = 2),
            nn.ELU(self.alpha, inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((64, 64))
        self.classifier = nn.Sequential(
            nn.Linear(32*64*64, 1024),
            nn.ELU(self.alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ELU(self.alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024),
            nn.ELU(self.alpha, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Xie2019(nn.Module):
    def __init__(self, num_classes, dropout_rate = 0.6):
        super(Xie2019, self).__init__()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            nn.Conv2d(3,64, 11, padding = 5, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, padding = 1, stride = 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class Alexnet_LRN(nn.Module):
    def __init__(self, num_classes, dropout_rate= 0.5):
        super(Alexnet_LRN, self).__init__()
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                        nn.ReLU(inplace=True),
                        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(64, 192, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                        nn.MaxPool2d(kernel_size=3, stride=2),
                        nn.Conv2d(192, 384, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(384, 384, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(384, 256, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=3, stride=2)
                        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
                          nn.Linear(256*6*6, 4096),
                          nn.ReLU(inplace=True),
                          nn.Dropout(self.dropout_rate),
                          nn.Linear(4096, 4096),
                          nn.ReLU(inplace=True),
                          nn.Dropout(self.dropout_rate),
                          nn.Linear(4096, num_classes)                        
        )


        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.features[4].bias, 1)
        nn.init.constant_(self.features[10].bias, 1)
        nn.init.constant_(self.features[12].bias, 1) 


        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight,mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x



def kumar2018(num_classes, pretrained = False, **kwargs):
    model = Kumar2018(num_classes, **kwargs)
    return model


def meijer2019(num_classes, pretrained = False, **kwargs):
    model = Meijer2019(num_classes, **kwargs)
    return model

    
def xie2019_binary(num_classes, pretrained = False, **kwargs):
    assert num_classes == 1
    model = Xie2019(num_classes, **kwargs)
    return model


def xie2019_multilabel(num_classes, pretrained = False, **kwargs):
    assert num_classes > 1
    model = Xie2019(1, **kwargs)

    model.classifier = nn.Sequential(
        nn.Linear(128*8*8, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(model.dropout_rate),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(model.dropout_rate),
        nn.Linear(512, num_classes))
    return model



def chen2018_binary(num_classes, pretrained = False, **kwargs): 
    assert num_classes == 1
    model = models.squeezenet1_0(num_classes = num_classes)
    return model


def chen2018_multilabel(num_classes, pretrained = False, binary_task = False, **kwargs):
    assert num_classes > 1
    model = models.inception_v3(num_classes = num_classes)
    return model


def hassan2019(num_classes, pretrained = False, **kwargs):
    model = models.alexnet(num_classes = num_classes)     
    return model



def hassan2019_custom(num_classes, pretrained = False, **kwargs):
    model = Alexnet_LRN(num_classes = num_classes)
    return model

