import os
import csv
from sklearn import metrics, cluster
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image
import argparse


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
from patchcore.utils import Matrix_Alpha_Unsupervised, Matrix_Alpha_Supervised
from patchcore.patchcore import AnomalyClusteringCore  # This is originated from PatchCore and it is modified a little bit.
import test


from munkres import Munkres
LOGGER = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

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
    # "data"
]


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


def feature_map_visualize(path,
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
    same_seeds(2023)
    # 加载数据集，dataloader
    train_dataset = mvtec.MVTecDataset(source=path, classname=category, resize=256, imagesize=224)
    test_dataset = mvtec.MVTecDataset(source=path, split=mvtec.DatasetSplit.TEST,
                                      classname=category, resize=256, imagesize=224)
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

    unloader = transforms.ToPILImage()
    label_current = 'start'
    for i in range(0, len(info), 1):
        info_i = info[i]
        Z_i = torch.mean(Z[i], dim=1)
        max_Z_i = max(Z_i)
        min_Z_i = min(Z_i)
        Z_i = Z_i.reshape(int(math.sqrt(len(Z_i))),
                          int(math.sqrt(len(Z_i)))).cpu().clone()
        Z_i = (Z_i - min_Z_i) / (max_Z_i - min_Z_i)
        # we clone the tensor to not do changes on it
        Z_i_PIL = unloader(Z_i)
        if label_current != info_i["anomaly"]:
            label_current = info_i["anomaly"]
            test.visualize(info_i, Z_i_PIL,
                           backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" +
                           str(target_embed_dimension) + "_" + "_".join(layers_to_extract_from))


def make_category_data(path,
                       category,
                       pretrain_embed_dimension,
                       target_embed_dimension,
                       backbone_names,
                       layers_to_extract_from,
                       patchsize,
                       save_path,
                       train_ratio=1.0,
                       tau=1,
                       supervised="unsupervised",
                       dataset="mvtec_ad"
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
    test_dataset = mvtec.MVTecDataset(source=path, split=mvtec.DatasetSplit.TEST,
                                      classname=category, resize=256, imagesize=224)
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

    info = []
    with tqdm(total=len(test_dataloader)) as progress:
        for image in test_dataloader:
            if isinstance(image, dict):
                with torch.no_grad():
                    del image['image']
                    del image['mask']
                    info.append(image)
            progress.update(1)
    # torch.save(info, "info/info_data.pickle")

    # 测试集embedding
    Z, label_test = anomalyclusteringcore_instance.embed(test_dataloader, supervised)
    Z = torch.tensor(Z).to(device)

    if supervised == "supervised":
        train_dataset = mvtec.MVTecDataset(source=path, classname=category, resize=256, imagesize=224)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        # 训练集embedding，计算权重
        Z_train, label_train = anomalyclusteringcore_instance.embed(train_dataloader, supervised)
        Z_train = torch.tensor(Z_train)
        Z_train = Z_train[:int(train_ratio * len(Z)), :, :].to(device)
        matrix_alpha = Matrix_Alpha_Supervised(tau=tau, k=1, Z=Z, Z_train=Z_train, device=device)

    elif supervised == "unsupervised":
        # 测试集计算权重
        matrix_alpha = Matrix_Alpha_Unsupervised(tau=tau,
                                                 k=1,
                                                 Z=Z,
                                                 device=device)
    else:
        matrix_alpha = torch.ones(Z.shape[0], Z.shape[1]) / Z.shape[1]

    # 加权embedding计算
    matrix_alpha = matrix_alpha.unsqueeze(1).float().to(device)

    X = np.array(torch.bmm(matrix_alpha, Z, out=None).squeeze(1).cpu())
    # 均值embedding计算
    # average_matrix = torch.ones(matrix_alpha.shape) / matrix_alpha.shape[2]
    # X_average = np.array(torch.bmm(average_matrix, Z, out=None).squeeze(1).cpu())
    # 存储为元组格式

    data_matrix = (matrix_alpha, X)
    os.makedirs(save_path + "/" + "_".join(layers_to_extract_from) + "_" + str(pretrain_embed_dimension) +
                "_" + str(target_embed_dimension) + "_" + str(float(tau)) + "_" + str(float(train_ratio))
                , exist_ok=True)
    # 存储权重矩阵与embedding
    torch.save(data_matrix, save_path + "/" + "_".join(layers_to_extract_from) + "_" +
               str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) + "_" + str(float(tau)) +
               "_" + str(float(train_ratio)) + "/matrix_alpha_X_" + category + "_" + supervised + ".pickle")
    print("{:-^80}".format(category + ' end'))
    return data_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Calculating Matrix on MVTec AD')
    parser.add_argument('--path', default='data/mvtec_ad',
                        type=str, help="Path to the dataset.")
    parser.add_argument('--backbone_names', nargs='+', default=["dino_vitbase8"], help='Architecture.')
    parser.add_argument('--layers_to_extract_from', nargs='+', default=["blocks.10", "blocks.11"])
    parser.add_argument('--pretrain_embed_dimension', default=2048, type=int, help='Pretrained Embedding Dimension')
    parser.add_argument('--target_embed_dimension', default=4096, type=int, help='Target Embedding Dimension')

    parser.add_argument('--output_dir', default="outputs", help='Path where to save segmentations')

    parser.add_argument("--patchsize", type=int, default=3, help="Patch Size.")
    parser.add_argument("--tau", type=float, default=1, help="Tau.")
    parser.add_argument("--train_ratio", type=float, default=1, help="The ratio of train data.")
    parser.add_argument('--supervised', default='unsupervised', type=str, help="Supervised or not")
    parser.add_argument('--dataset', default='mvtec_ad', type=str, help="Dataset to use.")
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # path = args.path
    path = args.path
    # 参数赋值
    pretrain_embed_dimension = args.pretrain_embed_dimension
    target_embed_dimension = args.target_embed_dimension
    backbone_names = args.backbone_names
    layers_to_extract_from = args.layers_to_extract_from
    patchsize = args.patchsize
    tau = args.tau
    supervised = args.supervised
    train_ratio = args.train_ratio
    dataset = args.dataset
    tau_list = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4, 8, 10, 12, 14, 18, 20]
    layer_list = [["layer1"], ["layer2"], ["layer3"], ["layer4"]]
    for supervised in ["unsupervised", "supervised"]:
        for tau in [tau]:
            train_ratio = train_ratio
            save_path = args.output_dir + "/" + dataset + "/" + backbone_names[0] + "/" + supervised
            os.makedirs(save_path, exist_ok=True)
            for category in _CLASSNAMES:
                data = make_category_data(path=path,
                                          category=category,
                                          pretrain_embed_dimension=pretrain_embed_dimension,
                                          target_embed_dimension=target_embed_dimension,
                                          backbone_names=backbone_names,
                                          layers_to_extract_from=layers_to_extract_from,
                                          patchsize=patchsize,
                                          save_path=save_path,
                                          tau=tau,
                                          train_ratio=train_ratio,
                                          supervised=supervised,
                                          dataset=dataset
                                          )



