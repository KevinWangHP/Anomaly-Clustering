import re
import matplotlib.pyplot as plt
import numpy as np
import os


def draw_metrics(category, metrics, supervised_res, unsupervised_res, directory):
    os.makedirs(directory, exist_ok=True)
    xlist = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4, 8, 10, 12, 14, 18, 20, "avg"]
    block_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, "norm"]
    plt.plot(xlist, supervised_res, label="supervised")
    plt.plot(xlist, unsupervised_res, label="unsupervised")
    plt.xlabel('Tau')
    plt.ylabel(metrics)
    plt.title(category + '-' + metrics)
    plt.legend()
    plt.savefig(directory + "/" + category + "_" + metrics)
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


def draw(pretrain_embed_dimension,
         target_embed_dimension,
         backbone_names,
         layers_to_extract_from,
         category):
    supervised_name = backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) \
                      + "_" + "_".join(layers_to_extract_from) + "_" + "supervised" + "_result.csv"
    unsupervised_name = backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) \
                      + "_" + "_".join(layers_to_extract_from) + "_" + "unsupervised" + "_result.csv"

    NMI_supervised, ARI_supervised, F1_supervised = read_file(category, supervised_name)
    NMI_unsupervised, ARI_unsupervised, F1_unsupervised = read_file(category, unsupervised_name)

    draw_metrics(category, "NMI", NMI_supervised, NMI_unsupervised,
                 "result/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension)
                 + "_" + "_".join(layers_to_extract_from))
    draw_metrics(category, "ARI", ARI_supervised, ARI_unsupervised,
                 "result/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension)
                 + "_" + "_".join(layers_to_extract_from))
    draw_metrics(category, "F1", F1_supervised, F1_unsupervised,
                 "result/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" + str(target_embed_dimension)
                 + "_" + "_".join(layers_to_extract_from))


if __name__ == '__main__':
    pretrain_embed_dimension = 2048
    target_embed_dimension = 4096
    backbone_names = ["dino_deitsmall8_300ep"]
    layers_to_extract_from = ["blocks.10", "blocks.11"]
    blocks_list = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5',
                   'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11',
                   'norm']

    _CLASSNAMES = [
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
        "carpet",
        "grid",
        "leather",
        "tile",
        "wood",
        "MVTec(object)",
        "MVTec(texture)"
    ]
    for category in _CLASSNAMES:
        draw(pretrain_embed_dimension=pretrain_embed_dimension,
             target_embed_dimension=target_embed_dimension,
             backbone_names=backbone_names,
             layers_to_extract_from=layers_to_extract_from,
             category=category)
