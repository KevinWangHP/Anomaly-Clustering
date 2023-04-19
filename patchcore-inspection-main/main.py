import os
from enum import Enum
import csv
from sklearn import metrics, cluster
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image

from scipy.spatial import distance
from torchvision import transforms
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.datasets.mvtec as mvtec
from munkres import Munkres
LOGGER = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

device = "cpu"

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AnomalyClusteringCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(AnomalyClusteringCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        print("{:-^80}".format("embedding"))
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            labels = []

            with tqdm(total=len(data)) as progress:
                for image in data:
                    if isinstance(image, dict):
                        is_anomaly = image["is_anomaly"]
                        image = image["image"]
                    with torch.no_grad():
                        input_image = image.to(torch.float).to(self.device)
                        features.append(self._embed(input_image))
                        labels.append(is_anomaly)
                    progress.update(1)
            return features, labels
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        # 添加3 * 3 AveragePooling
        features_new = []
        for feature in features:
            # feature = torch.nn.AvgPool2d(3, padding=1)(feature)
            if len(feature.shape) == 3:
                feature = feature[:, 1:, :]
                feature = feature.reshape(feature.shape[0],
                                          int(math.sqrt(feature.shape[1])),
                                          int(math.sqrt(feature.shape[1])),
                                          feature.shape[2])
                feature = feature.permute(0, 3, 1, 2)
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]])(feature)
            features_new.append(feature)
        features = features_new
        # 添加Unit L2 Norm
        del features_new


        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x


def Max_Hausdorff_Distance(i, j):
    return max(distance.directed_hausdorff(i, j), distance.directed_hausdorff(i, j))[0]


def Weight_Distance_Unsupervised(Z, i):
    matrix_min = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            matrix_min = torch.cat((matrix_min, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    return torch.mean(matrix_min, dim=1)


def Weight_Distance_Supervised(Z, Z_train, i):
    matrix_min = torch.tensor([]).to(device)
    for j in range(Z_train.shape[0]):
        dist_matrix = torch.cdist(Z[i], Z_train[j])
        dist_matrix = torch.min(dist_matrix, dim=1)[0].unsqueeze(1)
        matrix_min = torch.cat((matrix_min, dist_matrix), dim=1)
    matrix_min = torch.min(matrix_min, dim=1)[0]
    return matrix_min


def Matrix_Alpha_Unsupervised(tau, k, Z):
    print("{:-^80}".format("Calculating Unsupervised Alpha Matrix"))
    matrix_alpha = torch.tensor([]).to(device)
    with tqdm(total=int(Z.shape[0])) as progress:
        for i in range(Z.shape[0]):
            alpha_i = k * torch.exp(1 / tau * Weight_Distance_Unsupervised(Z, i).unsqueeze(0))
            alpha_i = alpha_i / alpha_i.sum()
            matrix_alpha = torch.cat((matrix_alpha, alpha_i), dim=0)
            progress.update(1)
    return matrix_alpha


def Matrix_Alpha_Supervised(tau, k, Z, Z_train, ratio):
    print("{:-^80}".format("Calculating Supervised Alpha Matrix"))
    Z_train = Z_train[:int(ratio * len(Z)), :, :]
    matrix_alpha = torch.tensor([]).to(device)
    with tqdm(total=int(Z.shape[0])) as progress:
        for i in range(Z.shape[0]):
            alpha_i = k * torch.exp(1 / tau * Weight_Distance_Supervised(Z, Z_train, i).unsqueeze(0))
            alpha_i = alpha_i / alpha_i.sum()
            matrix_alpha = torch.cat((matrix_alpha, alpha_i), dim=0)
            progress.update(1)
    return matrix_alpha


def make_category_data(path,
                       category,
                       pretrain_embed_dimension,
                       target_embed_dimension,
                       backbone_names,
                       layers_to_extract_from,
                       patchsize,
                       train_ratio=1,
                       tau=1,
                       supervised="unsupervised"
                       ):
    print("{:-^80}".format(category + ' start ' + supervised))
    # 参数初始化
    faiss_on_gpu = True
    faiss_num_workers = 4
    input_shape = (3, 224, 224)
    anomaly_scorer_num_nn = 5
    sampler = patchcore.sampler.IdentitySampler()
    backbone_seed = None
    backbone_name = backbone_names[0]

    loaded_patchcores = []

    # 加载数据集，dataloader
    train_dataset = mvtec.MVTecDataset(source=path, classname=category)
    test_dataset = mvtec.MVTecDataset(source=path, split=mvtec.DatasetSplit.TEST, classname=category)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]
    layers_to_extract_from = layers_to_extract_from_coll[0]

    if ".seed-" in backbone_name:
        backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
            backbone_name.split("-")[-1]
        )
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, backbone_seed

    nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

    # 实例化模型对象
    anomalyclusteringcore_instance = AnomalyClusteringCore(device)
    anomalyclusteringcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        device=device,
        input_shape=input_shape,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        featuresampler=sampler,
        anomaly_scorer_num_nn=anomaly_scorer_num_nn,
        nn_method=nn_method,
    )

    # 获取图片标签、path等信息
    info = []
    with tqdm(total=len(test_dataloader)) as progress:
        for image in test_dataloader:
            if isinstance(image, dict):
                with torch.no_grad():
                    del image['image']
                    del image['mask']
                    info.append(image)
            progress.update(1)

    # 测试集embedding
    Z, label_test = anomalyclusteringcore_instance.embed(test_dataloader)
    Z = torch.tensor(Z).to(device)
    if supervised == "supervised":
        # 训练集embedding，计算权重
        Z_train, label_train = anomalyclusteringcore_instance.embed(train_dataloader)
        Z_train = torch.tensor(Z_train).to(device)
        matrix_alpha = Matrix_Alpha_Supervised(tau=tau, k=1, Z=Z, Z_train=Z_train, ratio=train_ratio)

    elif supervised == "unsupervised":
        # 测试集计算权重
        matrix_alpha = Matrix_Alpha_Unsupervised(tau=tau,
                                                 k=1,
                                                 Z=Z)
    else:
        print("{:-^80}".format("Calculating Average Alpha Matrix"))
        matrix_alpha = torch.ones(Z.shape[0], Z.shape[1]) / Z.shape[1]


    # 加权embedding计算
    matrix_alpha = matrix_alpha.unsqueeze(1)
    X = np.array(torch.bmm(matrix_alpha, Z, out=None).squeeze(1))
    # 均值embedding计算
    # average_matrix = torch.ones(matrix_alpha.shape) / matrix_alpha.shape[2]
    # X_average = np.array(torch.bmm(average_matrix, Z, out=None).squeeze(1))
    # 存储为元组格式
    data_matrix = (matrix_alpha, X)

    # 存储权重矩阵与embedding
    torch.save(data_matrix, "out/"
               + backbone_name + "_" + str(pretrain_embed_dimension) + "_" +
               str(target_embed_dimension) + "_" + "_".join(layers_to_extract_from) + "_" +
               str(tau) + "_" + supervised + "/data_" + category + "_"
               + backbone_name + "_" + str(pretrain_embed_dimension) + "_" +
               str(target_embed_dimension) + "_" + "_".join(layers_to_extract_from) + "_" +
               str(tau) + "_" + supervised + ".pickle")
    print("{:-^80}".format(category + ' end'))
    return data_matrix


if __name__ == "__main__":
    path_local = "C:\\Users\\86155\\Desktop\\STUDY\\Graduate_design\\code\\mvtec_anomaly_detection"
    path = "/home/intern/code/mvtec_anomaly_detection"
    # 参数赋值
    pretrain_embed_dimension = 2048
    target_embed_dimension = 4096
    backbone_names = ["dino_deitsmall8_300ep"]
    layers_to_extract_from = ['blocks.10', 'blocks.11']
    patchsize = 3
    tau = 20
    supervised = "average"
    train_ratio = 1

    name = backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + \
           str(target_embed_dimension) + "_" + "_".join(layers_to_extract_from) + "_" + \
           str(tau) + "_" + supervised
    os.makedirs("out\\" + name, exist_ok=True)

    for category in _CLASSNAMES:
        data = make_category_data(path=path_local,
                                  category=category,
                                  pretrain_embed_dimension=pretrain_embed_dimension,
                                  target_embed_dimension=target_embed_dimension,
                                  backbone_names=backbone_names,
                                  layers_to_extract_from=layers_to_extract_from,
                                  patchsize=patchsize,
                                  tau=tau,
                                  train_ratio=train_ratio,
                                  supervised=supervised)





