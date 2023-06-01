import re
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_metrics(category, metrics, supervised_res, unsupervised_res,
                 supervised_res_vit, unsupervised_res_vit,
                 supervised_res_WRN50, unsupervised_res_WRN50,
                 average_res, directory):
    os.makedirs(directory, exist_ok=True)
    xlist = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4, 8, 10, 12, 14, 18, 20, "avg"]
    train_ratio_list = [i/10 for i in range(0, 14)]
    block_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    blocks_list = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    plt.plot(train_ratio_list, supervised_res, lw=1, c='blue', marker='s', ms=10, label="MVTec(object)")
    plt.plot(train_ratio_list, unsupervised_res, lw=1, c='orange', marker='o', ms=10, label="MVTec(texture)")
    # plt.plot(xlist, supervised_res_vit, lw=1, c='orange', marker='s', ms=5, label="VIT_supervised")
    # plt.plot(xlist, unsupervised_res_vit, lw=1, c='orange', marker='o', ms=5, label="VIT_unsupervised", linestyle='--')
    # plt.plot(xlist, supervised_res_WRN50[:len(xlist)], lw=1, c='red', marker='s', ms=5, label="WRN50_supervised")
    # plt.plot(xlist, unsupervised_res_WRN50[:len(xlist)], lw=1, c='red', marker='o', ms=5, label="WRN50_unsupervised", linestyle='--')
    # plt.plot(blocks_list, average_res, lw=1, c='red', marker='o', ms=10, label="average")
    plt.tick_params("x", labelsize=12)
    plt.tick_params("y", labelsize=20)
    # plt.plot(blocks_list, average_res, label="average")
    plt.axhline(supervised_res[0], lw=1, linestyle='--', c="red", label="unsupervised")
    plt.axhline(unsupervised_res[0], lw=1, linestyle='--', c="red", label="unsupervised")
    plt.xlabel('Train Ratio', fontsize=15)
    # plt.ylabel(metrics, fontsize=20, rotation=0)
    plt.title(category + '-' + metrics, fontsize=40)
    plt.legend(prop={'size': 10})
    plt.savefig("draft/图3-4/" + category + "_" + metrics)
    plt.show()


def read_file(category, file_name):
    file = open('result/' + file_name, 'r')
    # search the line including accuracy
    NMI = []
    ARI = []
    F1 = []
    for line in file:
        if category in line:
            Res = line.split(",")
            NMI.append(float(Res[1]))
            ARI.append(float(Res[2]))
            F1.append(float(Res[3]))
    file.close()
    return NMI, ARI, F1

def collect_data(backbone_name,
                 pretrain_embed_dimension,
                 target_embed_dimension,
                 layers_to_extract_from,
                 category):
    supervised_name = backbone_name+ "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) \
                      + "_" + "_".join(layers_to_extract_from) + "_" + "supervised" + "_train_ratio_result.csv"
    unsupervised_name = backbone_name + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) \
                        + "_" + "_".join(layers_to_extract_from) + "_" + "supervised" + "_train_ratio_result.csv"
    average_name = backbone_name + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) \
                   + "_" + "_".join(layers_to_extract_from) + "_" + "unsupervised" + "_result.csv"
    NMI_supervised, ARI_supervised, F1_supervised = read_file(category, supervised_name)
    NMI_unsupervised, ARI_unsupervised, F1_unsupervised = read_file(category, unsupervised_name)
    NMI_average, ARI_average, F1_average = read_file(category, average_name)
    return NMI_supervised, ARI_supervised, F1_supervised,\
           NMI_unsupervised, ARI_unsupervised, F1_unsupervised,\
           NMI_average, ARI_average, F1_average

def draw(pretrain_embed_dimension,
         target_embed_dimension,
         backbone_names,
         layers_to_extract_from,
         category):
    NMI_supervised, ARI_supervised, F1_supervised, \
    NMI_unsupervised, ARI_unsupervised, F1_unsupervised, \
    NMI_average, ARI_average, F1_average = collect_data(backbone_names[0],
                                                        pretrain_embed_dimension,
                                                        target_embed_dimension,
                                                        layers_to_extract_from,
                                                        category)

    NMI_supervised_vit, ARI_supervised_vit, F1_supervised_vit, \
    NMI_unsupervised_vit, ARI_unsupervised_vit, F1_unsupervised_vit, \
    NMI_average_vit, ARI_average_vit, F1_average_vit = collect_data(backbone_names[0],
                                                                    pretrain_embed_dimension,
                                                                    target_embed_dimension,
                                                                    layers_to_extract_from,
                                                                    "MVTec(texture)")

    NMI_supervised_WRN50, ARI_supervised_WRN50, F1_supervised_WRN50, \
    NMI_unsupervised_WRN50, ARI_unsupervised_WRN50, F1_unsupervised_WRN50, \
    NMI_average_WRN50, ARI_average_WRN50, F1_average_WRN50 = collect_data(backbone_names[0],
                                                                          pretrain_embed_dimension,
                                                                          target_embed_dimension,
                                                                          layers_to_extract_from,
                                                                          category)


    draw_metrics(category, "NMI", NMI_supervised, NMI_supervised_vit,
                 NMI_supervised_vit, NMI_unsupervised_vit,
                 NMI_supervised_WRN50, NMI_unsupervised_WRN50,
                 NMI_average,
                 "result/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension)
                 + "_" + "_".join(layers_to_extract_from))
    draw_metrics(category, "ARI", ARI_supervised, ARI_supervised_vit,
                 ARI_supervised_vit, ARI_unsupervised_vit,
                 ARI_supervised_WRN50, ARI_unsupervised_WRN50,
                 ARI_average,
                 "result/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension)
                 + "_" + "_".join(layers_to_extract_from))
    draw_metrics(category, "F1", F1_supervised, F1_supervised_vit,
                 F1_supervised_vit, F1_unsupervised_vit,
                 F1_supervised_WRN50, F1_unsupervised_WRN50,
                 F1_average,
                 "result/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension)
                 + "_" + "_".join(layers_to_extract_from))


if __name__ == '__main__':
    pretrain_embed_dimension = 2048
    target_embed_dimension = 4096
    backbone_names = ["wideresnet50"]
    layers_to_extract_from = ["blocks.10", "blocks.11"]
    blocks_list = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5',
                   'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11',
                   'norm']
    # blocks_list = ["layer1", "layer2", "layer3", "layer4", "avgpool"]

    _CLASSNAMES = [
        # "bottle",
        # "cable",
        # "capsule",
        # "hazelnut",
        # "metal_nut",
        # "pill",
        # "screw",
        # "toothbrush",
        # "transistor",
        # "zipper",
        # "carpet",
        # "grid",
        # "leather",
        # "tile",
        # "wood",
        "MVTec(object)",
        # "MVTec(texture)"
    ]
    for category in _CLASSNAMES:
        draw(pretrain_embed_dimension=pretrain_embed_dimension,
             target_embed_dimension=target_embed_dimension,
             backbone_names=backbone_names,
             layers_to_extract_from=["layer2", "layer3"],
             category=category)
