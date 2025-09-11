import argparse
import os
import sys
import time

from PIL import ImageDraw, ImageFont, Image
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

# from torchvision.datasets import CocoDetection
import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator

"""
CUDA_VISIBLE_DEVICES=0 \
python demo/test_ap_on_coco.py \
 -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
 -p /home/ashmal/Courses/CVS/Assignment_2/scripts/GroundingDINO/weights/groundingdino_swint_ogc.pth \
 --anno_path /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017/annotations/instances_val2017.json \
 --image_dir /home/ashmal/Courses/CVS/Assignment_2/data/2/coco2017/val2017
"""


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

        # import ipdb; ipdb.set_trace()

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # filt invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target


class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)  # 80, 256. normed

        id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                  41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

        # build a mapping from label_id to pos_map
        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (91, 256).T -> (bs, 100, 91)
        prob_to_label = prob_to_token @ pos_maps.T

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results

def box_iou_xyxy(boxes1, boxes2):
    """IoU between two sets of boxes in xyxy format."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def compute_prf1(all_predictions, cocoGt, iou_thr=0.5, score_thr=0.5):
    """Compute micro-averaged Precision/Recall/F1."""
    TP, FP, FN = 0, 0, 0
    preds_by_img = {}
    for p in all_predictions:
        if p["score"] < score_thr:
            continue
        preds_by_img.setdefault(p["image_id"], []).append(p)

    for img_id in cocoGt.getImgIds():
        ann_ids = cocoGt.getAnnIds(imgIds=[img_id], iscrowd=None)
        gts = cocoGt.loadAnns(ann_ids)
        gt_boxes = torch.tensor(
            [[g["bbox"][0], g["bbox"][1], g["bbox"][0]+g["bbox"][2], g["bbox"][1]+g["bbox"][3]] for g in gts],
            dtype=torch.float32
        )
        gt_labels = [g["category_id"] for g in gts]

        preds = preds_by_img.get(img_id, [])
        if len(preds) == 0 and len(gt_boxes) > 0:
            FN += len(gt_boxes)
            continue

        if len(preds) == 0:
            continue

        pred_boxes = torch.tensor([p["bbox"] for p in preds], dtype=torch.float32)
        pred_boxes[:, 2] += pred_boxes[:, 0]  # xywh -> xyxy
        pred_boxes[:, 3] += pred_boxes[:, 1]
        pred_labels = [p["category_id"] for p in preds]

        matched_gt = set()
        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou = 0.0
            best_j = -1
            for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if gl != pl or j in matched_gt:
                    continue
                iou = box_iou_xyxy(pb[None], gb[None])[0, 0].item()
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thr:
                TP += 1
                matched_gt.add(best_j)
            else:
                FP += 1
        FN += (len(gt_boxes) - len(matched_gt))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1, TP, FP, FN


def main(args):
    # config
    start_wall = time.time()
    model_time = 0.0
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)

    # if len(dataset) > 1000:
    #     dataset.ids = dataset.ids[:100]
    #     print(f"Using only {len(dataset)} images for evaluation")

    
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # run inference
    start = time.time()
    
    all_predictions = []
    for i, (images, targets) in enumerate(data_loader):
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # ---- measure model inference time ----
        loop_start = time.time()
        outputs = model(images, captions=input_captions)
        model_time += time.time() - loop_start
        # --------------------------------------

        # feed to the model
        outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)

        if i < 12:  # save first 12 images
            out_folder_path = "/home/ashmal/Courses/CVS/Assignment_2/scripts/Output/GroundingDINO/Sample_Outputs"
            img_id = targets[0]["image_id"]
            img_info = dataset.coco.loadImgs([img_id])[0]
            img_path = os.path.join(args.image_dir, img_info["file_name"])
            pil_img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.load_default()
        
            det = results[0]   # dict with 'boxes', 'scores', 'labels'
            for (x1, y1, x2, y2), score, label in zip(det["boxes"], det["scores"], det["labels"]):
                if score < 0.5:
                    continue
                cat_name = dataset.coco.cats[int(label.item())]["name"]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1), f"{cat_name}:{score:.2f}", fill="yellow", font=font)
        
            out_path = f"{out_folder_path}/groundingdino_{img_id}.jpg"
            pil_img.save(out_path)
            print("Saved visualization:", out_path)
    
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)


        # --- Save predictions in COCO-style format ---
        for target, output in zip(targets, results):
            image_id = int(target["image_id"])
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                all_predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })


        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())
    coco_stats = evaluator.coco_eval["bbox"].stats.tolist()
    mAP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl = coco_stats

    # --- Save predictions to file ---
    out_folder_path = "/home/ashmal/Courses/CVS/Assignment_2/scripts/Output/GroundingDINO/"
    out_json = os.path.join(out_folder_path, "groundingdino_test_predictions.json")
    with open(out_json, "w") as f:
        import json
        json.dump(all_predictions, f)
    print(f"Saved test predictions -> {out_json}")

    end_wall = time.time()
    wall_total = end_wall - start_wall
    num_imgs = len(dataset)
    
    eval_txt = os.path.join(out_folder_path, 'groundingdino_test_eval.txt')
    with open(eval_txt, 'w') as f:
        f.write("COCO Evaluation Metrics (pycocotools):\n")
        f.write(f"mAP@[.50:.95]: {mAP:.4f}\n")
        f.write(f"AP50: {AP50:.4f}\n")
        f.write(f"AP75: {AP75:.4f}\n")
        f.write(f"AP (small/medium/large): {APs:.4f}/{APm:.4f}/{APl:.4f}\n")
        f.write(f"AR@1/10/100: {AR1:.4f}/{AR10:.4f}/{AR100:.4f}\n")
        f.write(f"AR (small/medium/large): {ARs:.4f}/{ARm:.4f}/{ARl:.4f}\n\n")
    
        precision, recall, f1, TP, FP, FN = compute_prf1(all_predictions, dataset.coco, iou_thr=0.5, score_thr=0.5)

        f.write("Fixed-threshold PRF1 (IoU=0.50, score>=0.50):\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"(Counts) TP={TP} FP={FP} FN={FN}\n\n")
    
        f.write("Timing:\n")
        f.write(f"Images processed: {num_imgs}\n")
        f.write(f"Total wall time (s): {wall_total:.3f}\n")
        f.write(f"Model inference time total (s): {model_time:.3f}\n")
        f.write(f"Avg wall time per image (ms): {(wall_total/num_imgs)*1000:.3f}\n")
        f.write(f"Avg model time per image (ms): {(model_time/num_imgs)*1000:.3f}\n")
        f.write(f"Throughput wall (img/s): {num_imgs/wall_total:.3f}\n")
        f.write(f"Throughput model-only (img/s): {num_imgs/model_time}\n")
    
    print(f"Saved evaluation metrics -> {eval_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    args = parser.parse_args()

    main(args)
