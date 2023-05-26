# Anomaly Clustering
Please see Anomaly Clustering folder which is the code integration of the whole project.

Algorithm implementation using Pytorch for 
[Anomaly Clustering: Grouping Images into Coherent Clusters of Anomaly Types](https://openaccess.thecvf.com/content/WACV2023/html/Sohn_Anomaly_Clustering_Grouping_Images_Into_Coherent_Clusters_of_Anomaly_Types_WACV_2023_paper.html). 
Improve the algorithm with 
[DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) 
pretrained ViT. Implement algorithms based on 
[PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).


## Background

Anomaly detection is a subtask of computer vision, typically formulated as a binary classification problem. However, the
expressive power of binary classification labels is limited, and it is more practical to divide the data into multiple 
semantically coherent clusters. This paper reproduces the newly proposed 
[Anomaly Clustering](https://openaccess.thecvf.com/content/WACV2023/html/Sohn_Anomaly_Clustering_Grouping_Images_Into_Coherent_Clusters_of_Anomaly_Types_WACV_2023_paper.html) 
method, and proposes to use 
[DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) 
(self-distillation with no labels) method pre-trained ViT as a feature extractor for anomaly clustering to improve the 
clustering performance. The Anomaly Clustering method utilizes pre-trained image patch embeddings and traditional 
clustering methods to divide the data into coherent clusters of anomaly types. The method uses the Euclidean distance 
between weighted average embeddings as the distance function between images. The weights represent the importance of 
instances (i.e., image patches, which can highlight defective regions and they can be computed in an unsupervised way or
in a semi-supervised way when normal data is available. The DINO method employs label-free self-distillation, 
simplifying self-supervised training and enhancing the representation power of output feature maps by directly 
predicting the output of a teacher network constructed by momentum encoders using a standard cross-entropy loss. The 
model performs well on ImageNet. This paper conducts experiments on the defect detection dataset 
[MVTec AD](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html) 
to verify the effectiveness of the method. Compared to the method in the paper, using the DINO method pre-trained ViT 
as the backbone for image feature extraction improves clustering performance and achieves 
<font color=red>state-of-the-art</font> performance.


## Usage

First, put dataset in the folder `data/mvtec_ad`. Notice that the dataloader `mvtec.py` in the folder `datasets` is just
a copy of `models/datasets/mvtec.py`. When we run the program, we actually run into the original one.

###Calculate Matrix Alpha
To calculate the matrix alpha, use
>PYTHONPATH=models python examples/main.py --path data/mvtec_ad --backbone_names dino_vitbase8 --layers_to_extract_from blocks.10 blocks.11 --pretrain_embed_dimension 2048 --target_embed_dimension 4096 --output_dir outputs --patchsize 3 --tau 1 --train_ratio 1 --supervised unsupervised --dataset mvtec_ad

`path` is the path to the dataset. <br>
`backbone_names` are the backbone feature maps extractors. You can check the available backbones in 
`models/patchcore/backbones.py` <br>
`layers_to_extract_from` are the different layers used to fuse multiscale features. <br>
`pretrained_embed_dimension` is the target dimension of single layer features.<br>
`target_embed_dimension` is the target dimension of multi layer fused-features.<br>
`output_dir` is directory to save matrix alpha and X. <br>
`patchsize` is the n*n neighborhood of patch embeddings to fuse. <br>
`tau` controls the smoothness of matrix alpha. <br>
`train_ratio` is the ratio of training picture used in semi-supervised situation.<br>
`supervised` is the situation choosed from unsupervised, supervised and average.<br>
`dataset` is mvtec_ad.

Normally, we use a list of tau to calculate the matrix alpha. We change `for tau in [tau]:` to`for tau in tau_list:`.<br>
`tau_list:`is a list of tau.

###Calculate Metrics
To calculate the metrics of clustering: NMI, ARI, F1-micro. use 
>PYTHONPATH=models python examples/test.py

Normally, we calculate the results of different tau. However, when we are interested in the effect of other parameters, 
we can change `for tau in tau_list:` to other statement such as `for train_ratio in train_ratio_list:` 
or `for layer in layer_list:`.

###Draw Matrix Alpha
>PYTHONPATH=models python utils/draw_alpha.py

You must modify the parameters in the draw_alpha.py including <br>
`dataset` is the dataset name.<br>
`path_local` is the local path of the dataset<br>
`supervised` is choosed from unsupervised, supervised and average.<br>
`backbone_names` are the backbone feature maps extractors. <br>
`layers_to_extract_from` are the different layers used to fuse multiscale features. <br>
`pretrained_embed_dimension` is the target dimension of single layer features.<br>
`target_embed_dimension` is the target dimension of multi layer fused-features.<br>
`tau` controls the smoothness of matrix alpha. <br>
`train_ratio` is the ratio of training picture used in semi-supervised situation.<br>

## Result
| MVTec(object)  | Average |        |        | Unsupervised |        |        | Supervised |        |        |
|----------------|---------|--------|--------|--------------|--------|--------|------------|--------|--------|
| Metrics        | NMI     | ARI    | F1     | NMI          | ARI    | F1     | NMI        | ARI    | F1     |
| WideResNet50   | 0.310   | 0.188  | 0.434  |<font color=red>0.435</font>|<font color=red>0.305</font>|<font color=red>0.544</font>| 0.561      | 0.419  | 0.623  |
| ViT Base       | 0.350   | 0.241  | 0.477  | 0.318        | 0.154  | 0.448  | 0.459      | 0.305  | 0.567  |
| DINO ViT Base  | 0.372   | 0.227  | 0.485  | 0.430        | 0.292  | 0.543  | <font color=red>0.608</font> |<font color=red>0.496</font>|<font color=red>0.696</font>|
|                |         |        |        |              |        |        |            |        |        |

| MVTec(texture) | Average |        |        | Unsupervised |        |        | Supervised |        |        |
|----------------|---------|--------|--------|--------------|--------|--------|------------|--------|--------|
| Metrics        | NMI     | ARI    | F1     | NMI          | ARI    | F1     | NMI        | ARI    | F1     |
| WideResNet50   | 0.448   | 0.290  | 0.502  | 0.661        | 0.559  | 0.710  | 0.672      | 0.578  | 0.740  |
| ViT Base       | 0.685   | 0.610  | 0.736  | 0.648        | 0.569  | 0.728  | 0.727      | 0.654  | 0.786  |
| DINO ViT Base  | 0.635   | 0.551  | 0.696  |<font color=red>0.757</font>|<font color=red>0.686</font>| <font color=red>0.806</font> | <font color=red>0.790</font> |<font color=red>0.741</font>|<font color=red>0.857</font>|


## Related Efforts

- [PatchCore](https://github.com/amazon-science/patchcore-inspection) - Mainly based on PatchCore for code development.
- [DINO](https://github.com/facebookresearch/dino) -  Using DINO pretrained ViT to extract feature maps, and reached 
state-of-the-art result in Anomaly Clustering.

## Maintainers

[@KevinWangHP](https://github.com/KevinWangHP).


## License

This project is licensed under the Apache-2.0 License.
