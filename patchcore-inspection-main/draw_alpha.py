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

picture_index = "图3-10"
unloader = transforms.ToPILImage()
backbone_name = "wideresnet50"
pretrain_embed_dimension = 2048
target_embed_dimension = 2048
layers_to_extract_from = ["layer2", "layer3"]
layer_list = [["layer1"], ["layer2"],
              ["layer3"], ["layer4"]]
tau_list = [2]
tau = 2
supervised = "supervised"
visualize_list = ["bottle_broken_large", "capsule_poke", "carpet_color", "hazelnut_hole",
                   "leather_fold", "pill_crack", "tile_crack", "transistor_damaged_case",
                   "wood_hole", "zipper_split_teeth", "cable_cut_outer_insulation",
                   "toothbrush_defective", "grid_bent", "screw_scratch_head",
                   "metal_nut_scratch"]
# visualize_list = ["metal_nut_scratch", "metal_nut_bent", "metal_nut_color",
#                   "metal_nut_flip", "grid_broken", "grid_glue", "grid_bent",
#                   "grid_metal_contamination", "grid_thread"]
for train_ratio in range(20, 21):
    train_ratio = train_ratio / 10
    for layers_to_extract_from in layer_list:
        for category in _OBJECT+_TEXTURE:
            matrix_alpha_path = "out/mvtec_ad/" + backbone_name + "_" + \
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
                    if info_i["classname"][0] + "_" + label_current[0] in visualize_list:
                        path_local = "C:\\Users\\86155\\Desktop\\STUDY\\Graduate_design\\code\\mvtec_anomaly_detection"
                        path = "/home/intern/code/mvtec_anomaly_detection"
                        # 使用pillow库读取图片
                        fig = plt.figure(figsize=(4, 4), constrained_layout=True)
                        plt.axis('off')
                        process = transforms.Compose([transforms.Resize([256, 256]),
                                                      transforms.CenterCrop(224)])

                        img = Image.open(info_i["image_path"][0].replace(path, path_local)).convert("RGB")
                        img = process(img)
                        plt.imshow(img)
                        os.makedirs("draft\\" + picture_index, exist_ok=True)
                        fname = os.path.join("draft\\" + picture_index, ".0" + info_i["classname"][0] + "_" +
                                             info_i["anomaly"][0] + "_origin.png")
                        fig.savefig(fname)

                        if info_i["anomaly"][0] != "good":
                            fig1 = plt.figure(figsize=(4, 4), constrained_layout=True)
                            plt.axis('off')
                            img = Image.open(info_i["image_path"][0].replace("test", "ground_truth")
                                             .replace(".png", "_mask.png")
                                             .replace(path, path_local))
                            img = process(img)
                            plt.imshow(img, cmap='gray')
                            os.makedirs("draft\\" + picture_index, exist_ok=True)
                            fname = os.path.join("draft\\" + picture_index, ".1" + info_i["classname"][0] + "_" +
                                                 info_i["anomaly"][0] + "_mask.png")
                            fig1.savefig(fname)
                        fig2 = plt.figure(figsize=(4, 4), constrained_layout=True)
                        plt.axis('off')
                        plt.imshow(alpha_i_PIL)
                        os.makedirs("draft\\" + picture_index, exist_ok=True)

                        # if supervised == "unsupervised":
                        #     supervised = "a" + supervised
                        fname = os.path.join("draft\\" + picture_index,
                                             layers_to_extract_from[0] + "_" + str(int(tau)).rjust(2, '0') + "_" +
                                             info_i["classname"][0] + "_" + info_i["anomaly"][0] + ".png")
                        # if supervised == "aunsupervised":
                        #     supervised = "unsupervised"
                        fig2.savefig(fname)
                        print(f"{fname} saved.")