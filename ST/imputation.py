import os
import csv
import re
import time
import cv2
import pandas as pd
import numpy as np
import torch
from torch import nn
import json
import timm
from .util import *
from .contour_util import *
from .calculate_dis import *

# 加载模型
def load_model(model_path, config_path):
    """
    加载模型权重并构建模型。
    Args:
        model_path (str): 模型权重文件路径
        config_path (str): 模型配置文件路径
    Returns:
        nn.Module: 加载好的模型
        dict: 预训练配置
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    architecture = config_dict.get("architecture")
    if not architecture:
        raise ValueError("config.json must contain 'architecture' key to specify the model architecture.")

    if architecture != "vit_huge_patch14_224":
        raise ValueError(f"Expected architecture 'vit_huge_patch14_224', but got '{architecture}'.")

    model_args = config_dict.get("model_args", {})
    pretrained_cfg = config_dict.get("pretrained_cfg", {})

    model = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=model_args.get("num_classes", 0),
        img_size=model_args.get("img_size", 224),
        dynamic_img_size=model_args.get("dynamic_img_size", True),
        mlp_ratio=model_args.get("mlp_ratio", 5.3375),
        init_values=model_args.get("init_values", 1e-5),
        global_pool=model_args.get("global_pool", ""),
        reg_tokens=model_args.get("reg_tokens", 0),
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model_state_dict = model.state_dict()

    if "pos_embed" in state_dict:
        checkpoint_pos_embed = state_dict["pos_embed"]
        model_pos_embed = model_state_dict["pos_embed"]
        if checkpoint_pos_embed.shape[1] == 257 and model_pos_embed.shape[1] == 261:
            pos_embed_new = torch.zeros_like(model_pos_embed)
            pos_embed_new[:, :257, :] = checkpoint_pos_embed
            pos_embed_new[:, 257:, :] = checkpoint_pos_embed[:, -4:, :]
            state_dict["pos_embed"] = pos_embed_new

    for i in range(32):
        checkpoint_weight = state_dict.get(f"blocks.{i}.mlp.fc2.weight", None)
        if checkpoint_weight is not None and checkpoint_weight.shape == (1280, 3416):
            model_weight = model_state_dict[f"blocks.{i}.mlp.fc2.weight"]
            weight_new = torch.zeros_like(model_weight)
            weight_new[:, :3416] = checkpoint_weight
            weight_new[:, 3416:] = checkpoint_weight[:, -1:]
            state_dict[f"blocks.{i}.mlp.fc2.weight"] = weight_new

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, pretrained_cfg

# 提取图像特征的函数
def extract_embedding(x_pixel, y_pixel, image, model, beta, pretrained_cfg, device="cpu"):
    """
    使用深度学习模型提取图像区域的特征 embedding。
    """
    embeddings = []
    model = model.to(device)

    mean = np.array(pretrained_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = np.array(pretrained_cfg.get("std", [0.229, 0.224, 0.225]))
    img_size = pretrained_cfg.get("input_size", [3, 224, 224])[1]

    for x, y in zip(x_pixel, y_pixel):
        x_start = max(0, x - beta // 2)
        x_end = min(image.shape[0], x + beta // 2)
        y_start = max(0, y - beta // 2)
        y_end = min(image.shape[1], y + beta // 2)
        region = image[x_start:x_end, y_start:y_end]

        region = cv2.resize(region, (img_size, img_size))
        region = region / 255.0
        region = region.transpose(2, 0, 1)
        region = (region - mean[:, None, None]) / std[:, None, None]
        region = torch.tensor(region, dtype=torch.float32).unsqueeze(0)

        region = region.to(device)
        with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.float16):
            output = model(region)

        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        embedding = embedding.to(torch.float16).cpu().numpy()
        embeddings.append(embedding.flatten())

    return np.array(embeddings)

def imputation(img, raw, cnt, genes, shape="None", res=50, s=1, k=2, num_nbs=10):
    # 加载深度学习模型
    model_path = "/home/dingsq/dsq/Virchow2/pytorch_model.bin"
    config_path = "/home/dingsq/dsq/Virchow2/config.json"
    model, pretrained_cfg = load_model(model_path, config_path)

    # 创建二值掩码
    binary = np.zeros((img.shape[0:2]), dtype=np.uint8)
    cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)

    # 放大轮廓并创建二值掩码
    cnt_enlarged = scale_contour(cnt, 1.05)
    binary_enlarged = np.zeros(img.shape[0:2])
    cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)

    # 生成伪点坐标
    x_max, y_max = img.shape[0], img.shape[1]
    x_list = list(range(int(res), x_max, int(res)))
    y_list = list(range(int(res), y_max, int(res)))
    x = np.repeat(x_list, len(y_list)).tolist()
    y = y_list * len(x_list)
    sudo = pd.DataFrame({"x": x, "y": y})
    sudo = sudo[sudo.index.isin([i for i in sudo.index if (binary_enlarged[sudo.x[i], sudo.y[i]] != 0)])]
    sudo = sudo.reset_index(drop=True)

    # 提取伪点的特征嵌入
    b = res
    embeddings = extract_embedding(
        x_pixel=sudo.x.tolist(),
        y_pixel=sudo.y.tolist(),
        image=img,
        model=model,
        beta=b,
        pretrained_cfg=pretrained_cfg
    )

    # 创建 sudo_adata 并存储嵌入
    sudo_adata = AnnData(np.zeros((sudo.shape[0], len(genes))))
    sudo_adata.obs = sudo
    sudo_adata.var = pd.DataFrame(index=genes)
    sudo_adata.obsm["embedding"] = embeddings
    z_scale = np.max([np.std(sudo.x), np.std(sudo.y)]) * s
    sudo_adata.obs["z"] = np.mean(embeddings, axis=1) * z_scale  # 使用嵌入均值作为 z

    # ------------------------------------Known points---------------------------------#
    known_adata = raw[:, raw.var.index.isin(genes)]
    known_adata.obs["x"] = known_adata.obs["pixel_x"]
    known_adata.obs["y"] = known_adata.obs["pixel_y"]
    embeddings_known = extract_embedding(
        x_pixel=known_adata.obs["pixel_x"].astype(int).tolist(),
        y_pixel=known_adata.obs["pixel_y"].astype(int).tolist(),
        image=img,
        model=model,
        beta=b,
        pretrained_cfg=pretrained_cfg
    )
    known_adata.obsm["embedding"] = embeddings_known
    known_adata.obs["z"] = np.mean(embeddings_known, axis=1) * z_scale

    # -----------------------Distance matrix between sudo and known points-------------#
    start_time = time.time()
    dis = np.zeros((sudo_adata.shape[0], known_adata.shape[0]))
    x_sudo, y_sudo, z_sudo = sudo_adata.obs["x"].values, sudo_adata.obs["y"].values, sudo_adata.obs["z"].values
    x_known, y_known, z_known = known_adata.obs["x"].values, known_adata.obs["y"].values, known_adata.obs["z"].values
    print("Total number of sudo points: ", sudo_adata.shape[0])
    for i in range(sudo_adata.shape[0]):
        if i % 1000 == 0:
            print("Calculating spot", i)
        cord1 = np.array([x_sudo[i], y_sudo[i], z_sudo[i]])
        for j in range(known_adata.shape[0]):
            cord2 = np.array([x_known[j], y_known[j], z_known[j]])
            dis[i][j] = distance(cord1, cord2)
    print("--- %s seconds ---" % (time.time() - start_time))
    dis = pd.DataFrame(dis, index=sudo_adata.obs.index, columns=known_adata.obs.index)

    # -------------------------Fill gene expression using nbs---------------------------#
    for i in range(sudo_adata.shape[0]):
        if i % 1000 == 0:
            print("Imputing spot", i)
        index = sudo_adata.obs.index[i]
        dis_tmp = dis.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs]
        dis_tmp = (nbs.to_numpy() + 0.1) / np.min(nbs.to_numpy() + 0.1)  # avoid 0 distance
        if isinstance(k, int):
            weights = ((1 / (dis_tmp ** k)) / ((1 / (dis_tmp ** k)).sum()))
        else:
            weights = np.exp(-dis_tmp) / np.sum(np.exp(-dis_tmp))
        row_index = [known_adata.obs.index.get_loc(i) for i in nbs.index]
        sudo_adata.X[i, :] = np.dot(weights, known_adata.X[row_index, :])

    return sudo_adata