import torchvision.transforms as transforms
import torch
import math
import matplotlib.pyplot as plt
from PIL import Image
import os


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
unloader = transforms.ToPILImage()


dataset = "mvtec_ad"
path_local = "../data/" + dataset  # dataset local path

supervised = "supervised"
backbone_names = ["dino_vitbase8"]
pretrain_embed_dimension = 2048
target_embed_dimension = 4096
layers_to_extract_from = ["blocks.10", "blocks.11"]
tau = 2
train_ratio = 1
layer_list = [["layer1"], ["layer2"],
              ["layer3"], ["layer4"]]
blocks_list = [['blocks.0'], ['blocks.1'], ['blocks.2'], ['blocks.3'], ['blocks.4'], ['blocks.5'],
               ['blocks.6'], ['blocks.7'], ['blocks.8'], ['blocks.9'], ['blocks.10'], ['blocks.11'],
              ]
train_ratio_list = [0.1, 0.5, 1.0, 1.3]

# visualize category
visualize_list = ["bottle_broken_large", "capsule_poke", "carpet_color", "hazelnut_hole",
                   "leather_fold", "pill_crack", "tile_crack", "transistor_damaged_case",
                   "wood_hole", "zipper_split_teeth", "cable_cut_outer_insulation",
                   "toothbrush_defective", "grid_bent", "screw_scratch_head",
                   "metal_nut_scratch"]
# visualize_list = ["metal_nut_scratch", "metal_nut_bent", "metal_nut_color",
#                   "metal_nut_flip", "grid_broken", "grid_glue", "grid_bent",
#                   "grid_metal_contamination", "grid_thread"]


for train_ratio in [1]:
    for layers_to_extract_from in [layers_to_extract_from]:
        for category in _OBJECT + _TEXTURE:
            path = "/home/intern/code/mvtec_anomaly_detection"  # dataset server path, it is related to info_category.pickle.
            matrix_alpha_path = "../outputs/" + dataset + "/" + backbone_names[0] + "/" + supervised +\
                                "/" + "_".join(layers_to_extract_from) + "_" + str(pretrain_embed_dimension) +\
                                "_" + str(target_embed_dimension) + "_" + str(float(tau)) + "_" +\
                                str(float(train_ratio)) + "/matrix_alpha_X_" + category + "_" + supervised + ".pickle"

            matrix_alpha, X = torch.load(matrix_alpha_path, map_location='cpu')
            matrix_alpha = matrix_alpha.squeeze(1)
            info = torch.load("../outputs/" + dataset + "/info/info_" + category + ".pickle", map_location='cpu')
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
                    if info_i["classname"][0] + "_" + label_current[0] in visualize_list:

                        # 使用pillow库读取图片
                        fig = plt.figure(figsize=(4, 4), constrained_layout=True)
                        plt.axis('off')
                        process = transforms.Compose([transforms.Resize([256, 256]),
                                                      transforms.CenterCrop(224)])

                        img = Image.open(info_i["image_path"][0].replace(path, path_local)).convert("RGB")
                        img = process(img)
                        plt.imshow(img)
                        os.makedirs("../outputs/" + dataset + "/" + backbone_names[0] + "/" + supervised +
                                    "/" + "_".join(layers_to_extract_from) + "_" + str(pretrain_embed_dimension) +
                                    "_" + str(target_embed_dimension) + "_" + str(float(tau)) + "_" +
                                    str(float(train_ratio)) + "/picture", exist_ok=True)
                        fname = os.path.join("../outputs/" + dataset + "/" + backbone_names[0] + "/" + supervised +
                                             "/" + "_".join(layers_to_extract_from) + "_" + str(pretrain_embed_dimension)
                                             + "_" + str(target_embed_dimension) + "_" + str(float(tau)) + "_" +
                                             str(float(train_ratio)) + "/picture",
                                             info_i["classname"][0] + "_" + info_i["anomaly"][0] + "_origin.png")
                        fig.savefig(fname)

                        if info_i["anomaly"][0] != "good":
                            fig1 = plt.figure(figsize=(4, 4), constrained_layout=True)
                            plt.axis('off')
                            img = Image.open(info_i["image_path"][0].replace("test", "ground_truth")
                                             .replace(".png", "_mask.png")
                                             .replace(path, path_local))
                            img = process(img)
                            plt.imshow(img, cmap='gray')
                            fname = os.path.join("../outputs/" + dataset + "/" + backbone_names[0] + "/" + supervised +
                                                 "/" + "_".join(layers_to_extract_from) + "_" +
                                                 str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) +
                                                 "_" + str(float(tau)) + "_" + str(float(train_ratio)) + "/picture",
                                                 info_i["classname"][0] + "_" + info_i["anomaly"][0] + "_mask.png")
                            fig1.savefig(fname)
                        fig2 = plt.figure(figsize=(4, 4), constrained_layout=True)
                        plt.axis('off')
                        plt.imshow(alpha_i_PIL)

                        fname = os.path.join("../outputs/" + dataset + "/" + backbone_names[0] + "/" + supervised +
                                             "/" + "_".join(layers_to_extract_from) + "_" +
                                             str(pretrain_embed_dimension) + "_" + str(target_embed_dimension) +
                                             "_" + str(float(tau)) + "_" + str(float(train_ratio)) + "/picture",
                                             info_i["classname"][0] + "_" + info_i["anomaly"][0] + "_matrix.png")


                        fig2.savefig(fname)
                        print(f"{fname} saved.")