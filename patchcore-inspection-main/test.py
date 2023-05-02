import shutil

import torch
import math
import torchvision.transforms as transforms
import sklearn.cluster as cluster
from PIL import Image
from munkres import Munkres
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = '1'


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
    # "data"
]

_OBJECT = [
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
]

_TEXTURE = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize(info, alpha_PIL, name):
    path_local = "C:\\Users\\86155\\Desktop\\STUDY\\Graduate_design\\code\\mvtec_anomaly_detection"
    path = "/home/intern/code/mvtec_anomaly_detection"
    # 使用pillow库读取图片
    fig = plt.figure(figsize=(12, 4))
    process = transforms.Compose([transforms.Resize([256, 256]),
                                  transforms.CenterCrop(224)])

    img = Image.open(info["image_path"][0].replace(path, path_local)).convert("RGB")
    img = process(img)
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    if "mvtec_ad" in name:
        if info["anomaly"][0] != "good":
            img = Image.open(info["image_path"][0].replace("test", "ground_truth")
                             .replace(".png", "_mask.png")
                             .replace(path, path_local))
            img = process(img)
            ax2 = fig.add_subplot(132)
            ax2.imshow(img, cmap='gray')
        ax3 = fig.add_subplot(133)
        ax3.imshow(alpha_PIL)
        os.makedirs("out\\" + name + "\\visualize", exist_ok=True)

        fname = os.path.join("out\\" + name, "visualize", info["classname"][0] + "_" +
                             info["anomaly"][0] + ".png")
    else:
        ax3 = fig.add_subplot(133)
        ax3.imshow(alpha_PIL)
        os.makedirs("out\\" + name + "\\visualize", exist_ok=True)

        fname = os.path.join("out\\" + name, "visualize", info["classname"][0] + "_" + info["anomaly"][0] + "_" +
                             info["image_name"][0].split(".bmp")[0].split("\\")[-1] + ".png")
    plt.savefig(fname)
    print(f"{fname} saved.")
    # plt.show()


def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def calculate_metrics(category,
                      pretrain_embed_dimension,
                      target_embed_dimension,
                      backbone_names,
                      layers_to_extract_from,
                      patchsize,
                      train_ratio=1,
                      tau=0.1,
                      supervised="unsupervised",
                      dataset="mvtec_ad"):
    path_local = "C:\\Users\\86155\\Desktop\\STUDY\\Graduate_design\\code\\mvtec_anomaly_detection"
    path = "/home/intern/code/mvtec_anomaly_detection"


    unloader = transforms.ToPILImage()
    matrix_alpha_path = "out/" + dataset + "/" + backbone_names[0] + "_" + \
                        str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) + \
                        "_" + "_".join(layers_to_extract_from) + "_" + str(float(tau)) + "_" + \
                        supervised + "/data_" + category + "_" + supervised + ".pickle"

    matrix_alpha, X = torch.load(matrix_alpha_path, map_location='cpu')
    matrix_alpha = matrix_alpha.squeeze(1)
    info = torch.load("info/info_" + category + ".pickle", map_location='cpu')
    # 数据可视化
    label_current = 'start'
    for i in range(0, len(info), 1):
        info_i = info[i]
        max_alpha = max(matrix_alpha[i])
        alpha_i = matrix_alpha[i].reshape(int(math.sqrt(len(matrix_alpha[i]))),
                                          int(math.sqrt(len(matrix_alpha[i])))).cpu().clone()
        alpha_i = torch.nn.functional.interpolate(alpha_i.unsqueeze(0).unsqueeze(0), scale_factor=8,
                                                     mode="nearest")[0].cpu().numpy()
        # we clone the tensor to not do changes on it
        alpha_i_PIL = unloader(alpha_i/max_alpha)
        if label_current != info_i["anomaly"]:
        # if supervised != "average":
            label_current = info_i["anomaly"]
            visualize(info_i, alpha_i_PIL,
                      dataset + "/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" +
                      str(target_embed_dimension) + "_" + "_".join(layers_to_extract_from) + "_" +
                      str(float(tau)) + "_" + supervised)

    # 删除多标签实例
    X_one_category = np.zeros((1, X.shape[1]))
    label = []
    info_new = []
    for i in range(len(info)):
        if info[i]["anomaly"][0] != "combined":
            info_new.append(info[i])
            label.append(info[i]["anomaly"][0])
            X_one_category = np.append(X_one_category, np.expand_dims(X[i], axis=0), axis=0)
    X = X_one_category[1:]
    info = info_new
    del info_new
    del X_one_category

    le = LabelEncoder()
    label = le.fit_transform(label).astype(int)

    model = cluster.AgglomerativeClustering(n_clusters=len(set(label)))

    predict = model.fit_predict(X)
    predict = best_map(label, predict).astype(int)

    # 将图片分类到不同文件夹
    # for i in range(len(info)):
        # predict_cur = predict[i]
        # info_i = info[i]
        # old_file_path = info_i["image_path"][0].replace(path, path_local)
        # file_name = old_file_path.split("/")[-1]
        # new_file_path = os.path.join("out\\" + dataset + "/" + backbone_names[0] + "_" + str(pretrain_embed_dimension) + "_" +
        #                              str(target_embed_dimension) + "_" + "_".join(layers_to_extract_from) + "_" +
        #                              str(float(tau)) + "_" + supervised, info_i["classname"][0], str(predict_cur))
        # # 如果路径不存在，则创建
        # if not os.path.exists(new_file_path):
        #     os.makedirs(new_file_path)
        # # 新文件位置
        # new_file_path = os.path.join(new_file_path, info_i["anomaly"][0] + "_" + file_name)
        # print(str(i) + " 正在将 " + old_file_path + " 复制到 " + new_file_path)
        # # 复制文件
        # shutil.copyfile(old_file_path, new_file_path)



    NMI = metrics.normalized_mutual_info_score(label, predict)
    ARI = metrics.adjusted_rand_score(label, predict)
    F1 = metrics.f1_score(label, predict, average="micro")
    print(category)
    print(f'NMI: {NMI}')
    print(f'ARI: {ARI}')
    print(f'F1:{F1}\n')

    return NMI, ARI, F1, label, predict


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = '1'

    pretrain_embed_dimension = 2048
    target_embed_dimension = 2048
    backbone_names = ["dino_deitsmall8_300ep"]
    # backbone_names = ["vit_base"]

    layers_to_extract_from = ['blocks.10', 'blocks.11']
    patchsize = 3
    # tau_list = [0.2, 0.4, 0.8, 1, 1.5, 2, 3, 4, 8, 10, 12, 14, 18, 20]
    tau_list = [0.2, 0.4, 0.8, 1]
    tau = 2
    supervised = "unsupervised"

    import csv
    file_name = "result.csv"
    # 引用csv模块。
    csv_file = open('result.csv', 'w', newline='', encoding='gbk')
    # 调用open()函数打开csv文件，传入参数：文件名“demo.csv”、写入模式“w”、newline=''、encoding='gbk'
    writer = csv.writer(csv_file)
    # 用csv.writer()函数创建一个writer对象。
    writer.writerow([supervised])
    writer.writerow(["Category", "NMI", "ARI", "F1"])

    for i in range(12):
        layers_to_extract_from = ["blocks."+str(i)]
        writer.writerow(["---"] * 4)
        writer.writerow(["TAU="+str(tau)])
        NMI_OBJECT = 0
        ARI_OBJECT = 0
        F1_OBJECT = 0
        OBJECT_TOTAL = 0
        NMI_TEXTURE = 0
        ARI_TEXTURE = 0
        F1_TEXTURE = 0
        TEXTURE_TOTAL = 0
        for category in _OBJECT:
            print("{:-^80}".format(category))
            NMI, ARI, F1, label, predict = calculate_metrics(category=category,
                                                             pretrain_embed_dimension=pretrain_embed_dimension,
                                                             target_embed_dimension=target_embed_dimension,
                                                             backbone_names=backbone_names,
                                                             layers_to_extract_from=layers_to_extract_from,
                                                             patchsize=patchsize,
                                                             tau=tau,
                                                             supervised=supervised)
            writer.writerow([category, NMI, ARI, F1])
            NMI_OBJECT += NMI * len(label)
            ARI_OBJECT += ARI * len(label)
            F1_OBJECT += F1 * len(label)
            OBJECT_TOTAL += len(label)

        for category in _TEXTURE:
            print("{:-^80}".format(category))
            NMI, ARI, F1, label, predict = calculate_metrics(category=category,
                                                             pretrain_embed_dimension=pretrain_embed_dimension,
                                                             target_embed_dimension=target_embed_dimension,
                                                             backbone_names=backbone_names,
                                                             layers_to_extract_from=layers_to_extract_from,
                                                             patchsize=patchsize,
                                                             tau=tau,
                                                             supervised=supervised)

            writer.writerow([category, NMI, ARI, F1])
            NMI_TEXTURE += NMI * len(label)
            ARI_TEXTURE += ARI * len(label)
            F1_TEXTURE += F1 * len(label)
            TEXTURE_TOTAL += len(label)

        NMI_OBJECT /= OBJECT_TOTAL
        ARI_OBJECT /= OBJECT_TOTAL
        F1_OBJECT /= OBJECT_TOTAL
        print("MVTec(object)")
        print(f'NMI: {NMI_OBJECT}')
        print(f'ARI: {ARI_OBJECT}')
        print(f'F1:{F1_OBJECT}\n')
        writer.writerow(["MVTec(object)", NMI_OBJECT, ARI_OBJECT, F1_OBJECT])

        NMI_TEXTURE /= TEXTURE_TOTAL
        ARI_TEXTURE /= TEXTURE_TOTAL
        F1_TEXTURE /= TEXTURE_TOTAL
        print("MVTec(texture)")
        print(f'NMI: {NMI_TEXTURE}')
        print(f'ARI: {ARI_TEXTURE}')
        print(f'F1:{F1_TEXTURE}\n')
        writer.writerow(["MVTec(texture)", NMI_TEXTURE, ARI_TEXTURE, F1_TEXTURE])

    # for category in _CLASSNAMES:
    #     print("{:-^80}".format(category))
    #     tau_list = [4.0]
    #     for tau in tau_list:
    #         NMI, ARI, F1, label, predict = calculate_metrics(category=category,
    #                                                          pretrain_embed_dimension=pretrain_embed_dimension,
    #                                                          target_embed_dimension=target_embed_dimension,
    #                                                          backbone_names=backbone_names,
    #                                                          layers_to_extract_from=layers_to_extract_from,
    #                                                          patchsize=patchsize,
    #                                                          tau=tau,
    #                                                          supervised=supervised,
    #                                                          dataset="newData")
    #
    #         writer.writerow([str(tau), NMI, ARI, F1])

    csv_file.close()

    # 关闭文件


