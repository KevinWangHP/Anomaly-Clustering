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


def visualize(info, alpha_PIL):
    path_local = "C:\\Users\\86155\\Desktop\\STUDY\\Graduate_design\\code\\mvtec_anomaly_detection"
    path = "/home/intern/code/mvtec_anomaly_detection"
    # 使用pillow库读取图片
    fig = plt.figure(figsize=(12, 4))
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224)])

    img = Image.open(info["image_path"][0].replace(path, path_local))
    img = process(img)
    ax1 = fig.add_subplot(131)
    ax1.imshow(img)
    if info["anomaly"][0] != "good":
        img = Image.open(info["image_path"][0].replace("test", "ground_truth")
                         .replace(".png", "_mask.png")
                         .replace(path, path_local))
        img = process(img)
        ax2 = fig.add_subplot(132)
        ax2.imshow(img, cmap='gray')
    ax3 = fig.add_subplot(133)
    ax3.imshow(alpha_PIL)
    plt.show()


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


def calculate_metrics(category, average):
    unloader = transforms.ToPILImage()
    unsupervised = "_unsupervised.pickle"
    supervised = "_supervised.pickle"
    info, matrix_alpha, Z_list = torch.load("tmp/data_" + category + supervised, map_location='cpu')

    # 数据可视化
    label_current = 'start'
    for i in range(0, len(info), 1):
        info_i = info[i]
        max_alpha = max(matrix_alpha[i])
        alpha_i = matrix_alpha[i].reshape(int(math.sqrt(len(matrix_alpha[i]))),
                                          int(math.sqrt(len(matrix_alpha[i])))).cpu().clone()
        # we clone the tensor to not do changes on it
        alpha_i_PIL = unloader(alpha_i/max_alpha)
        if label_current != info_i["anomaly"]:
            label_current = info_i["anomaly"]
            visualize(info_i, alpha_i_PIL)

    if average:
        matrix_alpha = torch.ones(matrix_alpha.shape) / matrix_alpha.shape[2]
    else:
        matrix_alpha = matrix_alpha.unsqueeze(1)

    X = np.array(torch.bmm(matrix_alpha, Z_list, out=None).squeeze(1))

    # 删除多标签实例
    X_one_category = np.zeros((1, X.shape[1]))
    label = []
    for i in range(len(info)):
        if info[i]["anomaly"][0] != "combined":
            label.append(info[i]["anomaly"][0])
            X_one_category = np.append(X_one_category, np.expand_dims(X[i], axis=0), axis=0)
    X = X_one_category[1:]
    del X_one_category

    le = LabelEncoder()
    label = le.fit_transform(label)

    model = cluster.AgglomerativeClustering(n_clusters=len(set(label)))

    predict = model.fit_predict(X)
    predict = best_map(label, predict)

    NMI = metrics.normalized_mutual_info_score(label, predict)
    ARI = metrics.adjusted_rand_score(label, predict)
    F1 = metrics.f1_score(label, predict, average="micro")
    print("Weighted Average")
    print(f'NMI: {NMI}')
    print(f'ARI: {ARI}')
    print(f'F1:{F1}\n')

    return NMI, ARI, F1


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = '1'

    import csv

    # 引用csv模块。
    csv_file = open('result.csv', 'w', newline='', encoding='gbk')
    # 调用open()函数打开csv文件，传入参数：文件名“demo.csv”、写入模式“w”、newline=''、encoding='gbk'
    writer = csv.writer(csv_file)
    # 用csv.writer()函数创建一个writer对象。
    writer.writerow(["Category", "NMI", "ARI", "F1"])
    for i in _CLASSNAMES:
        print("{:-^80}".format(i))
        NMI, ARI, F1 = calculate_metrics(category=i, average=True)
        writer.writerow([i, NMI, ARI, F1])
    csv_file.close()
    # 关闭文件


