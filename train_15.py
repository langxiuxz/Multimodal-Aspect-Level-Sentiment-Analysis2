import os
import gc
import random
import time
import json
import math
import warnings
import argparse
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, RobertaModel, BertModel
from transformers import AdamW


import argparse

from catr.models import caption
from catr.datasets import coco, utils
from catr.configuration import Config

from model.MabsaModel import *
from model.EmoGNN import gnn_data

from utils.helpers import *
from utils.data_utils import *
from utils.dataset import *



import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image


# Config
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--data_dir', type=str, default="./absa_data")
parser.add_argument('--catr_ckpt', type=str, default="catr/checkpoint.pth")
parser.add_argument('--result_dir', type=str, default="result")
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--dataset', type=str, default="twitter2017") # twitter2015 or twitter2017
parser.add_argument('--model', type=str, default="bertweet-base") # "bert-base-uncased" "vinai/bertweet-base"

parser.add_argument('--num_cycles', type=float, default=0.5)
parser.add_argument('--num_warmup_steps', type=int, default=0)
parser.add_argument('--adamw_correct_bias', type=bool, default=True)
parser.add_argument('--scheduler', type=str, default='linear') # ['linear', 'cosine']
parser.add_argument('--print_freq', type=int, default=30)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--max_grad_norm', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--lr_catr', type=float, default=2e-5)
parser.add_argument('--lr_backbone', type=float, default=5e-6)
parser.add_argument('--max_caption_len', type=int, default=12)
parser.add_argument('--max_len', type=int, default=72)
parser.add_argument('--sample_k', type=int, default=10)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=4)

args = parser.parse_args()

device = torch.device(args.device)
LOGGER = get_logger(os.path.join(args.log_dir, args.dataset))
# train_df, val_df,  test_df = load_data(args)


# 加载Mask R-CNN模型
mask_rcnn = maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn.eval().to(device)

# # 使用Mask R-CNN对图像进行实例分割并提取ROIs
# def segment_and_crop(image_path, score_threshold=0.5):
#     """
#     使用Mask R-CNN对输入图像进行实例分割并裁剪。
#     返回裁剪后的感兴趣区域（ROIs）。
#     """
#     # 加载图像
#     image = Image.open(image_path).convert("RGB")
#     image_tensor = F.to_tensor(image).to(device).unsqueeze(0)

#     # 获取实例分割结果
#     with torch.no_grad():
#         predictions = mask_rcnn(image_tensor)

#     rois = []
#     masks = predictions[0]['masks'].cpu().numpy()
#     boxes = predictions[0]['boxes'].cpu().numpy()
#     scores = predictions[0]['scores'].cpu().numpy()

#     # 筛选高置信度的检测结果并裁剪
#     for i in range(len(scores)):
#         if scores[i] >= score_threshold:
#             mask = masks[i, 0]
#             bbox = boxes[i].astype(int)
#             x1, y1, x2, y2 = bbox
#             cropped = np.array(image)[y1:y2, x1:x2] * mask[y1:y2, x1:x2, None]
#             cropped_image = Image.fromarray((cropped * 255).astype(np.uint8))
#             rois.append(cropped_image)


#     return rois
def segment_and_filter(image_path, low_score_threshold=0.2):
    """
    使用Mask R-CNN对输入图像进行实例分割。
    删除低置信度（低于一定比例）的区域，保留高置信度的完整图像内容。
    
    参数:
        image_path (str): 输入图像的路径。
        low_score_threshold (float): 删除低置信度区域的置信度阈值，默认值为0.2。
    
    返回:
        Image: 经过处理的图像，仅保留高置信度区域的内容。
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device).unsqueeze(0)

    # 获取实例分割结果
    with torch.no_grad():
        predictions = mask_rcnn(image_tensor)

    # 提取预测结果
    masks = predictions[0]['masks'].cpu().numpy()  # 提取掩码
    boxes = predictions[0]['boxes'].cpu().numpy()  # 提取边界框
    scores = predictions[0]['scores'].cpu().numpy()  # 提取置信度分数

    # 删除低置信度区域
    keep_indices = np.where(scores >= low_score_threshold)[0]  # 仅保留高置信度区域
    # final_mask = np.zeros_like(masks[0, 0])  # 初始化图像掩码
    if len(keep_indices) == 0:
        #print(f"No high-confidence objects detected in image: {image_path}. Returning original image.")
        return image  # 如果没有高置信度区域，直接返回原始图像
    # 初始化图像掩码
    final_mask = np.zeros_like(masks[0, 0])

    # 合并所有高置信度区域的掩码
    for idx in keep_indices:
        final_mask = np.maximum(final_mask, masks[idx, 0])

    # 应用最终掩码到原图
    filtered_image = np.array(image) * final_mask[:, :, None]
    filtered_image = Image.fromarray((filtered_image * 255).astype(np.uint8))

    return filtered_image


def preprocess_with_segmentation(data_df, args):
    """
    对数据集中的图像进行实例分割和裁剪，返回更新后的DataFrame。
    """
    updated_images = []

    # 构造图像路径
    for image_id in data_df['image_id']:
        image_path = os.path.join(args.data_dir, args.dataset + "_images", image_id)  
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 使用实例分割
        # rois = segment_and_crop(image_path)
        filtered_image = segment_and_filter(image_path)
        updated_images.append(filtered_image)
        # if len(rois) > 0:
        #     updated_images.append(rois[0])  # 选择第一个ROI作为主要对象
        # else:
        #     updated_images.append(Image.open(image_path).convert("RGB"))  # 如果没有ROI，使用原图

    data_df['processed_images'] = updated_images  # 添加裁剪后的图像到DataFrame
    return data_df




# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
args.tokenizer = tokenizer
args.pad_token_id = args.tokenizer.pad_token_id
args.end_token_id = args.tokenizer.sep_token_id

## Train fn
def train_fn(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch):
    model = model.train()
    losses = []
    correct_predictions = 0
    start = end = time.time()
    for step, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        image = d["image"]
        image_mask = d["image_mask"]
        caption = d["caption"].to(device)
        cap_mask = d["caption_mask"].to(device)
        labels = d["labels"].to(device)
        input_ids_tt = d["input_ids_tt"].to(device)
        attention_mask_tt = d["attention_mask_tt"].to(device)
        input_ids_at = d["input_ids_at"].to(device)
        attention_mask_at = d["attention_mask_at"].to(device)
        samples = utils.NestedTensor(image, image_mask).to(device)
        if not isinstance(samples, utils.NestedTensor):
            raise TypeError("Unsupported type:", type(samples))
        outputs = model(samples, caption, cap_mask,
                input_ids, attention_mask, loss_fn,
                input_ids_tt, attention_mask_tt, input_ids_at, attention_mask_at)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels).item()
        losses.append(loss.item())
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # clip grad
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        end = time.time()
        if step % args.print_freq == 0 or step == (len(data_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                'Elapsed {remain:s} '
                        'Loss: {loss:.4f} '
                        'Grad: {grad_norm:.4f}  '
                        'LR: {lr:.8f}  '
                        .format(epoch+1, step, len(data_loader),
                                remain=timeSince(start, float(step+1)/len(data_loader)),
                                loss=loss.item(),
                                grad_norm=grad_norm,
                                lr=scheduler.get_lr()[0]))

    return correct_predictions / n_examples, np.mean(losses)

# eval fn
def format_eval_output(rows):
    tweets, targets, labels, predictions = zip(*rows)
    tweets = np.vstack(tweets)
    targets = np.vstack(targets)
    labels = np.vstack(labels)
    predictions = np.vstack(predictions)
    results_df = pd.DataFrame()
    results_df["tweet"] = tweets.reshape(-1).tolist()
    results_df["target"] = targets.reshape(-1).tolist()
    results_df["label"] = labels
    results_df["prediction"] = predictions
    return results_df


def eval_model(model, data_loader, loss_fn, device, n_examples, detailed_results=False):
    model = model.eval()
    losses = []
    correct_predictions = 0
    rows = []
    with torch.no_grad():
      for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        image = d["image"]
        image_mask = d["image_mask"]
        caption = d["caption"].to(device)
        cap_mask = d["caption_mask"].to(device)
        labels = d["labels"].to(device)
        samples = utils.NestedTensor(image, image_mask).to(device)
        input_ids_tt = d["input_ids_tt"].to(device)
        attention_mask_tt = d["attention_mask_tt"].to(device)
        input_ids_at = d["input_ids_at"].to(device)
        attention_mask_at = d["attention_mask_at"].to(device)
        outputs = model(samples, caption ,cap_mask,
                    input_ids, attention_mask, loss_fn,
                    input_ids_tt, attention_mask_tt, input_ids_at, attention_mask_at,is_training=False)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        correct_predictions += torch.sum(preds == labels).item()
        losses.append(loss.item())
        rows.extend(
          zip(d["review_text"],
            d["targets"],
            d["labels"].numpy(),
            preds.cpu().numpy(),
          )
        )

      if detailed_results:
          return (correct_predictions / n_examples,
                np.mean(losses),
                format_eval_output(rows),
            )

    return correct_predictions / n_examples, np.mean(losses)

def train_loop():
    LOGGER.info(f"========== Start Training ==========")
#/////////
    # 使用实例分割对图像数据进行裁剪
    LOGGER.info(f"========== Performing Instance Segmentation ==========")
    train_df, val_df,  test_df = load_data(args)
    train_df = preprocess_with_segmentation(train_df, args)
    val_df = preprocess_with_segmentation(val_df, args)
    test_df = preprocess_with_segmentation(test_df, args)
#///////////
    # Create DataLoader
    image_captions = None
    train_dataset = TwitterDataset(args, train_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    test_dataset = TwitterDataset(args, test_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    val_dataset = TwitterDataset(args, val_df, image_captions, coco.val_transform, utils.nested_tensor_from_tensor_list)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # Model
    config = Config()
    config.vocab_size = 64000
    config.pad_token_id = args.pad_token_id
    catr, _ = caption.build_model(config)
    checkpoint = torch.load(args.catr_ckpt)
    catr.to(device)
    catr.load_state_dict(checkpoint['model'])
    model = CustomModel(args, catr, gnn_data)
    model.to(device)
    # Configure the optimizer and scheduler.
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "catr" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,},
        {"params": [p for n, p in model.named_parameters() if "catr" in n and 'backbone' not in n and p.requires_grad],
        "lr": args.lr_catr,},
    ]
    optimizer = AdamW(param_dicts, lr=args.lr, correct_bias=args.adamw_correct_bias)

    num_train_steps = int(len(train_df)/args.batch_size*args.epochs)
    scheduler = get_scheduler(args, optimizer, num_train_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.epochs):
        print(f"===============Epoch {epoch + 1}/{args.epochs}==============")
        start_time = time.time()

        train_acc, train_loss = train_fn(
        model, train_loader, loss_fn, optimizer, device, scheduler, len(train_df), epoch
        )

        val_acc, val_loss, dr = eval_model(model, val_loader, loss_fn, device, len(val_df), detailed_results=True)
        macro_f1 = f1_score(dr.label, dr.prediction, average="macro")
        LOGGER.info(f'Epoch {epoch+1} - Val loss {val_loss} accuracy {val_acc} macro f1 {macro_f1}')

        test_acc, test_loss, dr = eval_model(model, test_loader, loss_fn, device, len(test_df), detailed_results=True)
        macro_f1 = f1_score(dr.label, dr.prediction, average="macro")
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - Train loss {train_loss} accuracy {train_acc}" time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Test loss {test_loss} accuracy {test_acc} macro f1 {macro_f1}')
    
    torch.cuda.empty_cache()
    gc.collect()
    LOGGER.info(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")
    return dr

if __name__ == "__main__":
    seed_everything(seed = args.seed)
    x = train_loop()


