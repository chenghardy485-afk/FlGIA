import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import timm
import os
from .models_vit import vit_small_patch16, vit_tiny_patch16
from .rkd_hypernetwork import rkdLinear


class ModelViT(nn.Module):
    def __init__(self, args):
        super(ModelViT, self).__init__()
        if args.vit_type == 'vit_small':
            self.feature_extractor = timm.create_model("vit_small_patch16_384", pretrained=True)
        elif args.vit_type == 'vit_tiny':
            self.feature_extractor = timm.create_model("vit_tiny_patch16_384", pretrained=True)
        # self.feature_extractor = timm.create_model("vit_small_patch16_384", pretrained=True)
        # fixed base model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.feature_extractor.head.in_features, args.num_classes)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x):
        x = self.feature_extractor.forward_features(x)
        x = x[:, 0]  # class token
        y = self.fc(x)
        return x, y


class AdapterHyperNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vit_type, reduction_factor=16):
        super(AdapterHyperNet, self).__init__()

        layers = [nn.Linear(embedding_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)
        if vit_type == 'vit_small':
            self.input_dim = 384
        elif vit_type == 'vit_tiny':
            self.input_dim = 192
        # self.input_dim = input_dim
        self.down_sample_size = self.input_dim // reduction_factor
        # generate parameters
        self.down_sampler_weights = nn.ModuleList(
            [nn.Linear(hidden_dim, self.input_dim * self.down_sample_size) for _ in range(12)])
        self.down_sampler_bias = nn.ModuleList([nn.Linear(hidden_dim, self.down_sample_size) for _ in range(12)])
        self.up_sampler_weights = nn.ModuleList(
            [nn.Linear(hidden_dim, self.input_dim * self.down_sample_size) for _ in range(12)])
        self.up_sampler_bias = nn.ModuleList([nn.Linear(hidden_dim, self.input_dim) for _ in range(12)])
        self.norm_weight = nn.ModuleList([nn.Linear(hidden_dim, self.input_dim) for _ in range(12)])
        self.norm_bias = nn.ModuleList([nn.Linear(hidden_dim, self.input_dim) for _ in range(12)])

    def forward(self, client_embedding):
        features = self.mlp(client_embedding)
        weights = {}
        for i in range(12):
            keys = 'blocks.' + str(i) + '.adapter.'
            weights[keys + "down_sampler.weight"] = self.down_sampler_weights[i](features).view(self.down_sample_size,
                                                                                                self.input_dim)
            weights[keys + "down_sampler.bias"] = self.down_sampler_bias[i](features).view(-1)
            weights[keys + "up_sampler.weight"] = self.up_sampler_weights[i](features).view(self.input_dim,
                                                                                            self.down_sample_size)
            weights[keys + "up_sampler.bias"] = self.up_sampler_bias[i](features).view(-1)
            weights[keys + "norm.weight"] = self.norm_weight[i](features).view(-1)
            weights[keys + "norm.bias"] = self.norm_bias[i](features).view(-1)
        return weights

class rkd_AdapterHyperNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vit_type, reduction_factor=16):
        super(rkd_AdapterHyperNet, self).__init__()

        layers = [nn.Linear(embedding_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)
        if vit_type == 'vit_small':
            self.input_dim = 384
        elif vit_type == 'vit_tiny':
            self.input_dim = 192
        # self.input_dim = input_dim
        self.down_sample_size = self.input_dim // reduction_factor
        self.rkd_d1 = self.input_dim // 4
        self.rkd_d2 = self.input_dim // 4
        self.rkd_d3 = self.rkd_d1 // 16
        self.rkd_d4 = self.down_sample_size // self.rkd_d3
        self.rkd_d5 = self.down_sample_size
        self.rkd_d6 = reduction_factor
        # generate parameters
        self.down_sampler_weights = nn.ModuleList(
            [rkdLinear(hidden_dim, self.rkd_d1, self.rkd_d2) for _ in range(12)])
        self.down_sampler_bias = nn.ModuleList([rkdLinear(hidden_dim, self.rkd_d3, self.rkd_d4) for _ in range(12)])
        self.up_sampler_weights = nn.ModuleList(
            [rkdLinear(hidden_dim, self.rkd_d1, self.rkd_d2) for _ in range(12)])
        self.up_sampler_bias = nn.ModuleList([rkdLinear(hidden_dim, self.rkd_d5, self.rkd_d6) for _ in range(12)])
        self.norm_weight = nn.ModuleList([rkdLinear(hidden_dim, self.rkd_d5, self.rkd_d6) for _ in range(12)])
        self.norm_bias = nn.ModuleList([rkdLinear(hidden_dim, self.rkd_d5, self.rkd_d6) for _ in range(12)])

    def forward(self, client_embedding):
        features = self.mlp(client_embedding)
        weights = {}
        for i in range(12):
            keys = 'blocks.' + str(i) + '.adapter.'
            weights[keys + "down_sampler.weight"] = self.down_sampler_weights[i](features).view(self.down_sample_size,
                                                                                                self.input_dim)
            weights[keys + "down_sampler.bias"] = self.down_sampler_bias[i](features).view(-1)
            weights[keys + "up_sampler.weight"] = self.up_sampler_weights[i](features).view(self.input_dim,
                                                                                            self.down_sample_size)
            weights[keys + "up_sampler.bias"] = self.up_sampler_bias[i](features).view(-1)
            weights[keys + "norm.weight"] = self.norm_weight[i](features).view(-1)
            weights[keys + "norm.bias"] = self.norm_bias[i](features).view(-1)
        return weights


class ModelViT_Hyper(nn.Module):
    def __init__(self, args):
        super(ModelViT_Hyper, self).__init__()
        if args.vit_type == 'vit_small':
            self.feature_extractor = vit_small_patch16()  # feature_extractor
            self.feature_extractor.load_state_dict(timm.create_model("vit_small_patch16_384", pretrained=True).state_dict(),
                                                    strict=False)
        elif args.vit_type == 'vit_tiny':
            self.feature_extractor = vit_tiny_patch16()  # feature_extractor
            self.feature_extractor.load_state_dict(timm.create_model("vit_tiny_patch16_384", pretrained=True).state_dict(),
                                                    strict=False)        
        # self.feature_extractor = vit_small_patch16()  # feature_extractor
        # self.feature_extractor.load_state_dict(timm.create_model("vit_small_patch16_384", pretrained=True).state_dict(),
        #                                           strict=False)
        for name, param in self.feature_extractor.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        self.fc = nn.Linear(self.feature_extractor.head.in_features, args.num_classes)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

        # client embedding
        self.client_embedding = nn.Embedding(num_embeddings=1, embedding_dim=args.embed_dim)
        # hypernetwork
        if   args.train_rule == 'HyperFL-LPM':
             self.hypernetwork = AdapterHyperNet(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim, vit_type=args.vit_type)
        elif args.train_rule == 'HyperFL++-LPM':
             self.hypernetwork = rkd_AdapterHyperNet(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim, vit_type=args.vit_type)
        # self.hypernetwork = AdapterHyperNet(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim)

    def forward(self, x):
        # generate weights
        client_embedding = self.client_embedding(torch.tensor(0).cuda())
        weights = self.hypernetwork(client_embedding)
        self.feature_extractor.load_state_dict(weights, strict=False)  # only load adapter weights

        x = self.feature_extractor.forward_features(x)
        x = x[:, 0]  # class token
        y = self.fc(x)
        return x, y

    def predict(self, x):
        x = self.feature_extractor.forward_features(x)
        x = x[:, 0]  # class token
        y = self.fc(x)
        return y

    def generate_weight(self):
        client_embedding = self.client_embedding(torch.tensor(0).cuda())
        weights = self.hypernetwork(client_embedding)
        return weights


class ModelViT_Adapter(nn.Module):
    def __init__(self, args):
        super(ModelViT_Adapter, self).__init__()
        if args.vit_type == 'vit_small':
            self.feature_extractor = vit_small_patch16()  # feature_extractor
            self.feature_extractor.load_state_dict(timm.create_model("vit_small_patch16_384", pretrained=True).state_dict(),
                                                    strict=False)
        elif args.vit_type == 'vit_tiny':
            self.feature_extractor = vit_tiny_patch16()  # feature_extractor
            self.feature_extractor.load_state_dict(timm.create_model("vit_tiny_patch16_384", pretrained=True).state_dict(),
                                                    strict=False)
        # self.feature_extractor = vit_small_patch16()  # feature_extractor
        # self.feature_extractor.load_state_dict(timm.create_model("vit_small_patch16_384", pretrained=True).state_dict(),
        #                                        strict=False)
        for name, param in self.feature_extractor.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        self.fc = nn.Linear(self.feature_extractor.head.in_features, args.num_classes)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x):
        x = self.feature_extractor.forward_features(x)
        x = x[:, 0]  # class token
        y = self.fc(x)
        return x, y

    def predict(self, x):
        x = self.feature_extractor.forward_features(x)
        x = x[:, 0]  # class token
        y = self.fc(x)
        return y
