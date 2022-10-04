import numpy as np
import torch
import torch.nn as nn
import torchvision.models as torch_models
from .graph_layers import GraphConvolution, ConvGraphCombination



class ResNet_GCNN(nn.Module):
    def __init__(self, backbone, num_classes, word_embeddings, adj_mat, lrelu_slope = 0.2):
        super(ResNet_GCNN, self).__init__()

        self.backbone = backbone
        self.backbone.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.backbone.fc = nn.Identity()

        self.lrelu_slope = lrelu_slope

        self.gc1 = GraphConvolution(num_classes, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.lrelu = nn.LeakyReLU(self.lrelu_slope, inplace=True)

        assert word_embeddings.shape[0] == adj_mat.shape[0], "amount of word embeddings does not match amount of classes in adjacency matrix: Word Emb {} - Adj Mat {}".format(word_embeddings.shape, adj_mat.shape)
        assert word_embeddings.shape[0] == num_classes, "amount of word embeddings does not match amount of classes provided: Word Emb {} - # Classes {}".format(word_embeddings.shape, num_classes)
        assert adj_mat.shape[0] == adj_mat.shape[1], "the adjacency matrix is not square: Adj Mat {}".format( adj_mat.shape)

        self.register_buffer("adj", adj_mat)
        self.register_buffer("word_embeddings", word_embeddings)


    def forward(self, x):

        _adj = self.adj.detach()
        _words = self.word_embeddings.detach()

        # CNN Module
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        # GCN Module
        gcn_features = self.gc1(_words, _adj)
        gcn_features = self.lrelu(gcn_features)
        gcn_features = self.gc2(gcn_features, _adj)
        gcn_features = self.lrelu(gcn_features)

        gcn_features = gcn_features.transpose(0,1)
        
        return torch.matmul(x, gcn_features) # logits


class ResNet_KSSNET(nn.Module):
    def __init__(self, backbone, num_classes, word_embeddings, adj_mat, lrelu_slope = 0.2):
        super(ResNet_KSSNET, self).__init__()

        self.backbone = backbone
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone.fc = nn.Identity()

        self.lrelu_slope = lrelu_slope

        self.gc1 = GraphConvolution(num_classes, 256)
        self.gc2 = GraphConvolution(256, 512)
        self.gc3 = GraphConvolution(512, 1024)
        self.gc4 = GraphConvolution(1024, 2048)
        self.lrelu = nn.LeakyReLU(self.lrelu_slope, inplace=True)
        
        self.convgcn1 = ConvGraphCombination(nn.Conv2d(num_classes, 256, kernel_size=1, stride=1, bias=False), nn.Tanh())
        self.convgcn2 = ConvGraphCombination(nn.Conv2d(num_classes, 512, kernel_size=1, stride=1, bias=False), nn.Tanh())
        self.convgcn3 = ConvGraphCombination(nn.Conv2d(num_classes, 1024, kernel_size=1, stride=1, bias=False), nn.Tanh())

        assert word_embeddings.shape[0] == adj_mat.shape[0], "amount of word embeddings does not match amount of classes in adjacency matrix: Word Emb {} - Adj Mat {}".format(word_embeddings.shape, adj_mat.shape)
        assert word_embeddings.shape[0] == num_classes, "amount of word embeddings does not match amount of classes provided: Word Emb {} - # Classes {}".format(word_embeddings.shape, num_classes)
        assert adj_mat.shape[0] == adj_mat.shape[1], "the adjacency matrix is not square: Adj Mat {}".format( adj_mat.shape)

        self.register_buffer("adj", adj_mat)
        self.register_buffer("word_embeddings", word_embeddings)


    def forward(self, x):

        _adj = self.adj.detach()
        _words = self.word_embeddings.detach()

        # CNN Module
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        gcn = self.gc1(_words, _adj)
        x = self.convgcn1(x, gcn)
        gcn = self.lrelu(gcn)

        x = self.backbone.layer2(x)
        gcn = self.gc2(gcn, _adj)
        x = self.convgcn2(x, gcn)
        gcn = self.lrelu(gcn)

        x = self.backbone.layer3(x)
        gcn = self.gc3(gcn, _adj)
        x = self.convgcn3(x, gcn)
        gcn = self.lrelu(gcn)

        x = self.backbone.layer4(x)
        gcn = self.gc4(gcn, _adj)
        gcn = self.lrelu(gcn)
        gcn = gcn.transpose(0,1)

        x = self.backbone.avgpool(x)
        x = x.view(x.shape[0], -1)

        return torch.matmul(x, gcn) # Logits





def _resnetgcn():
    backbone = torch_models.resnet101(num_classes=1)
    word_embeddings = np.eye(17)
    adj_mat = np.random.randn(17, 17)

    word_embeddings = torch.from_numpy(word_embeddings).float()
    adj_mat = torch.from_numpy(adj_mat).float()

    data = torch.rand(5, 3, 224, 224)

    model = ResNet_GCNN(backbone, 17, word_embeddings, adj_mat)

    model(data)

    return model

    
def _resnetkssnet():
    backbone = torch_models.resnet101(num_classes=1)
    word_embeddings = np.eye(17)
    adj_mat = np.random.randn(17, 17)

    word_embeddings = torch.from_numpy(word_embeddings).float()
    adj_mat = torch.from_numpy(adj_mat).float()

    data = torch.rand(5, 3, 224, 224)

    model = ResNet_KSSNET(backbone, 17, word_embeddings, adj_mat)

    model(data)

    return model


def parameter_count(model, name):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("{} - {}/{}".format(name, pytorch_total_trainparams, pytorch_total_params))

    return pytorch_total_trainparams, pytorch_total_params



if __name__ == "__main__":
    
    mlgcn = _resnetgcn()
    kssnet = _resnetkssnet()

    resnet101 = torch_models.resnet101(num_classes=1)


    base_train, base_total = parameter_count(resnet101, "ResNet101")
    mlgcn_train, mlgcn_total = parameter_count(mlgcn, "MLGCN-ResNet101")
    kss_train, kss_total = parameter_count(kssnet, "KSSNET-ResNet101")

    print(mlgcn_train/base_train, mlgcn_train-base_train, mlgcn_total/base_total, mlgcn_total-base_total)
    print(kss_train/base_train, kss_train-base_train, kss_total/base_total, kss_total-base_total)
