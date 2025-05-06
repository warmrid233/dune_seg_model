import os
import pathlib
from typing import Dict, Optional

import cv2
import gradio as gr
import torch
import torch.nn.functional as F

from scribbleprompt.models.network import UNet
from scribbleprompt.models.unet import rescale_inputs, prepare_inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RES = 512

file_dir = pathlib.Path(os.path.dirname(__file__))
# example_dir = file_dir / "examples"
#
# test_examples = [str(example_dir / x) for x in sorted(os.listdir(example_dir)) if not x.endswith('.npy')]
# default_example = test_examples[0]

# exp_dir = file_dir / "../checkpoints" 
# exp_dir = file_dir / "../output/20250425_192419-2DJJ-3d32feba0e20cc8c5b6049c08e8a235d/checkpoints"
exp_dir = file_dir / "../output/20250427_180601-SMP7-217b4438f3df3345fb2e409bb3e60e52/checkpoints"
default_model = 'ScribblePrompt-Unet'

model_dict = {
    # 'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt'
    'ScribblePrompt-Unet': 'max-val_od-dice_score.pt'
    # 'ScribblePrompt-Unet': 'epoch-400.pt'
}


# -----------------------------------------------------------------------------
# Model Predictor Class
# -----------------------------------------------------------------------------

class Predictor:
    """
    Wrapper for ScribblePrompt-UNet model
    """

    def __init__(self, path: str, verbose: bool = True):

        assert path.exists(), f"Checkpoint {path} does not exist"

        self.path = path
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.load()
        self.model.eval()
        self.to_device()

    def build_model(self):
        """
        Build the model
        """
        self.model = UNet(
            in_channels=5,
            out_channels=1,
            features=[192, 192, 192, 192],
        )

    def load(self):
        """
        Load the state of the model from a checkpoint file.
        """
        with (self.path).open("rb") as f:
            state = torch.load(f, map_location=self.device)
            self.model.load_state_dict(state["model"], strict=True)
            if self.verbose:
                print(
                    f"Loaded checkpoint from {self.path} to {self.device}"
                )

    def to_device(self):
        """
        Move the model to cpu or gpu
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def predict(self, prompts: Dict[str, any], img_features: Optional[torch.Tensor] = None,
                multimask_mode: bool = False):
        """
        Make predictions!

        Returns:
            mask (torch.Tensor): H x W
            img_features (torch.Tensor): B x 1 x H x W (for SAM models)
            low_res_mask (torch.Tensor): B x 1 x H x W logits
        """
        if self.verbose:
            print("point_coords", prompts.get("point_coords", None))
            print("point_labels", prompts.get("point_labels", None))
            print("box", prompts.get("box", None))
            print("img", prompts.get("img").shape, prompts.get("img").min(), prompts.get("img").max())
            if prompts.get("scribbles") is not None:
                print("scribbles", prompts.get("scribbles", None).shape, prompts.get("scribbles").min(),
                      prompts.get("scribbles").max())

        original_shape = prompts.get('img').shape[-2:]

        # Rescale to 128 x 128
        prompts = rescale_inputs(prompts)

        # Prepare inputs for ScribblePrompt unet (1 x 5 x 128 x 128)
        x = prepare_inputs(prompts).float()

        with torch.no_grad():
            yhat = self.model(x.to(self.device)).cpu()

        mask = torch.sigmoid(yhat)

        # Resize for app resolution
        mask = F.interpolate(mask, size=original_shape, mode='bilinear').squeeze()

        # mask: H x W, yhat: 1 x 1 x H x W
        return mask, None, yhat


# -----------------------------------------------------------------------------
# Model initialization functions
# -----------------------------------------------------------------------------

def load_model(exp_key: str = default_model):
    """
    exp_key: ScribblePrompt
    model_dict = {
    # 'ScribblePrompt-Unet': 'ScribblePrompt_unet_v1_nf192_res128.pt'
    'ScribblePrompt-Unet': 'max-val_od-dice_score.pt'
    # 'ScribblePrompt-Unet': 'epoch-400.pt'
    }
    """
    fpath = exp_dir / model_dict.get(exp_key)
    exp = Predictor(fpath)
    return exp, None


# -----------------------------------------------------------------------------
# Vizualization functions
# -----------------------------------------------------------------------------

def _get_overlay(img, lay, const_color="l_blue"):
    """
    Helper function for preparing overlay
    """
    assert lay.ndim == 2, "Overlay must be 2D, got shape: " + str(lay.shape)

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    assert img.ndim == 3, "Image must be 3D, got shape: " + str(img.shape)

    if const_color == "blue":
        const_color = 255 * np.array([0, 0, 1])
    elif const_color == "green":
        const_color = 255 * np.array([0, 1, 0])
    elif const_color == "red":
        const_color = 255 * np.array([1, 0, 0])
    elif const_color == "l_blue":
        const_color = np.array([31, 119, 180])
    elif const_color == "orange":
        const_color = np.array([255, 127, 14])
    else:
        raise NotImplementedError

    x, y = np.nonzero(lay)
    for i in range(img.shape[-1]):
        img[x, y, i] = const_color[i]

    return img


def image_overlay(img, mask=None, scribbles=None, contour=False, alpha=0.5):
    """
    Overlay the ground truth mask and scribbles on the image if provided
    """
    # assert img.ndim == 2, "Image must be 2D, got shape: " + str(img.shape)
    # output = np.repeat(img[...,None], 3, axis=-1)
    #######
    if img.ndim == 2:
        output = np.repeat(img[..., None], 3, axis=-1).astype(np.uint8)
    elif img.ndim == 3 and img.shape[-1] == 3:
        output = img.copy()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    #######
    if mask is not None:

        assert mask.ndim == 2, "Mask must be 2D, got shape: " + str(mask.shape)

        if contour:
            contours = cv2.findContours((mask[..., None] > 0.5).astype(np.uint8), cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours[0], -1, (0, 255, 0), 1)
        else:
            mask_overlay = _get_overlay(img, mask)
            mask2 = 0.5 * np.repeat(mask[..., None], 3, axis=-1)
            output = cv2.convertScaleAbs(mask_overlay * mask2 + output * (1 - mask2))

    if scribbles is not None:
        pos_scribble_overlay = _get_overlay(output, scribbles[0, ...], const_color="green")
        cv2.addWeighted(pos_scribble_overlay, alpha, output, 1 - alpha, 0, output)
        neg_scribble_overlay = _get_overlay(output, scribbles[1, ...], const_color="red")
        cv2.addWeighted(neg_scribble_overlay, alpha, output, 1 - alpha, 0, output)

    return output


def viz_pred_mask(img, mask=None, point_coords=None, point_labels=None, bbox_coords=None, seperate_scribble_masks=None,
                  binary=True):
    """
    Visualize image with clicks, scribbles, predicted mask overlaid
    """
    assert isinstance(img, np.ndarray), "Image must be numpy array, got type: " + str(type(img))
    #######
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1).astype(np.uint8)
    else:
        img = img.copy()
    #######
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

    if binary and mask is not None:
        mask = 1 * (mask > 0.5)

    out = image_overlay(img, mask=mask, scribbles=seperate_scribble_masks)

    if point_coords is not None:
        for i, (col, row) in enumerate(point_coords):
            if point_labels[i] == 1:
                cv2.circle(out, (col, row), 2, (0, 255, 0), -1)
            else:
                cv2.circle(out, (col, row), 2, (255, 0, 0), -1)

    if bbox_coords is not None:
        for i in range(len(bbox_coords) // 2):
            cv2.rectangle(out, bbox_coords[2 * i], bbox_coords[2 * i + 1], (255, 165, 0), 1)
        if len(bbox_coords) % 2 == 1:
            cv2.circle(out, tuple(bbox_coords[-1]), 2, (255, 165, 0), -1)
    return out


# -----------------------------------------------------------------------------
# Collect scribbles
# -----------------------------------------------------------------------------

def get_scribbles(seperate_scribble_masks, last_scribble_mask, scribble_img, label: int):
    """
    Record scribbles
    """
    assert isinstance(seperate_scribble_masks, np.ndarray), \
        "seperate_scribble_masks must be numpy array, got type: " + str(type(seperate_scribble_masks))

    if scribble_img is not None:

        color_mask = scribble_img.get('mask')
        scribble_mask = color_mask[..., 0] / 255

        not_same = (scribble_mask != last_scribble_mask)
        if not isinstance(not_same, bool):
            not_same = not_same.any()

        if not_same:
            # In case any scribbles were removed
            corrected_scribble_masks = np.stack(2 * [(scribble_mask > 0)], axis=0) * seperate_scribble_masks
            corrected_last_scribble_mask = last_scribble_mask * (scribble_mask > 0)

            delta = (scribble_mask - corrected_last_scribble_mask) > 0
            new_scribbles = scribble_mask * delta
            corrected_scribble_masks[label, ...] = np.clip(corrected_scribble_masks[label, ...] + new_scribbles,
                                                           a_min=0, a_max=1)

            last_scribble_mask = scribble_mask
            seperate_scribble_masks = corrected_scribble_masks

        return seperate_scribble_masks, last_scribble_mask


def get_predictions(predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks,
                    low_res_mask, img_features, multimask_mode):
    """
    Make predictions
    """
    box = None
    if len(bbox_coords) == 1:
        gr.Error("Please click a second time to define the bounding box")
        box = None
    elif len(bbox_coords) == 2:
        box = torch.Tensor(bbox_coords).flatten()[None, None, ...].int().to(device)  # B x n x 4

    if seperate_scribble_masks is not None:
        scribble = torch.from_numpy(seperate_scribble_masks)[None, ...].to(device)
    else:
        scribble = None
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        input_img = np.mean(input_img, axis=-1).astype(np.uint8)
    prompts = dict(
        img=torch.from_numpy(input_img)[None, None, ...].to(device) / 255,
        point_coords=torch.Tensor([click_coords]).int().to(device) if len(click_coords) > 0 else None,
        point_labels=torch.Tensor([click_labels]).int().to(device) if len(click_labels) > 0 else None,
        scribbles=scribble,
        mask_input=low_res_mask.to(device) if low_res_mask is not None else None,
        box=box,
    )

    mask, img_features, low_res_mask = predictor.predict(prompts, img_features, multimask_mode=multimask_mode)

    return mask, img_features, low_res_mask


def refresh_predictions(predictor, input_img, click_coords, click_labels, bbox_coords, current_label,
                        scribble_img, seperate_scribble_masks, last_scribble_mask,
                        low_res_mask, img_features, binary_checkbox, multimask_mode):
    """
    input_img: [512,512,3] 原图
    output_img： [
        {
            'name':'/home/jiangzheng/tmp/gradio/b7684713780b4fba817b7266f8cee9d2f24b4814/image.png'
            'data':'http://127.0.0.1:8024/file=/home/jiangzheng/tmp/gradio/b7684713780b4fba817b7266f8cee9d2f24b4814/image.png'
            'is_file':True
        },
         {
            'name':'/home/jiangzheng/tmp/gradio/b7684713780b4fba817b7266f8cee9d2f24b4814/image.png'
            'data':'http://127.0.0.1:8024/file=/home/jiangzheng/tmp/gradio/b7684713780b4fba817b7266f8cee9d2f24b4814/image.png'
            'is_file':True
        },
    ]
    click_coords: [【x,y】]
    click_labels: [1,1]
    bbox_coords: str: "Positive (green)" / "Negative (red)"
    scribble_img: dict {
        "image": [512,512,3] 【0，255】# 上一次原图和mask调和的图像
        "mask": [512,512,3] 【0，255】 # 当前涂鸦的所有痕迹
    }
    seperate_scribble_masks: [2,512,512] 涂鸦痕迹，正反两个通道 绿色+红色 【0/1】，表示历史scribble的痕迹
    last_scribble_mask: [512,512]【0/1】，表示涂鸦历史痕迹（合并seperate_scribble_masks）
    best_mask: None [512, 512]
    low_res_mask: None [1,1,128,128]
    img_features: None 
    binary_checkbox: bool Ture
    multimask_mode: True
    """
    # Record any new scribbles
    # visualize_binary_mask(seperate_scribble_masks, "seperate_scribble_masks")
    # visualize_binary_mask(last_scribble_mask, "last_scribble_mask")
    # visualize_rgb_image_and_mask(scribble_img['image'], scribble_img['mask'], "scribble_img")
    seperate_scribble_masks, last_scribble_mask = get_scribbles(
        seperate_scribble_masks, last_scribble_mask, scribble_img,
        label=current_label
    )
    # Make prediction
    best_mask, img_features, low_res_mask = get_predictions(
        predictor, input_img, click_coords, click_labels, bbox_coords, seperate_scribble_masks, low_res_mask,
        img_features, multimask_mode
    )

    # Update input visualizations
    mask_to_viz = best_mask.numpy()
    click_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords,
                                    seperate_scribble_masks, binary_checkbox)
    scribble_input_viz = viz_pred_mask(input_img, mask_to_viz, click_coords, click_labels, bbox_coords, None,
                                       binary_checkbox)

    out_viz = [
        viz_pred_mask(input_img, mask_to_viz, point_coords=None, point_labels=None, bbox_coords=None,
                      seperate_scribble_masks=None, binary=binary_checkbox),
        255 * (mask_to_viz[..., None].repeat(axis=2, repeats=3) > 0.5) if binary_checkbox else mask_to_viz[
            ..., None].repeat(axis=2, repeats=3),
    ]

    click_input_viz = {
        'image': click_input_viz,
        'mask': scribble_img.get("mask") if scribble_img.get("mask") is not None else np.zeros((RES, RES, 3), dtype=np.uint32)
    }
    scribble_input_viz = {
        'image': scribble_input_viz,
        'mask': scribble_img.get("mask") if scribble_img.get("mask") is not None else np.zeros((RES, RES, 3), dtype=np.uint32)
    }
    # out_viz = {
    #     'image': out_viz,
    #     'mask': scribble_img.get("mask") if scribble_img.get("mask") is not None else np.zeros((RES, RES, 3), dtype=np.uint32)
    # }
    return (click_input_viz, scribble_input_viz, out_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks,
            last_scribble_mask)


def get_select_coords(predictor, input_img, best_mask, low_res_mask,
                      click_coords, click_labels, bbox_coords,
                      seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
                      output_img, binary_checkbox, multimask_mode, autopredict_checkbox, current_coords, current_label):
    """
    Record user click and update the prediction
    """
    # Record click coordinates
    # if bbox_label:
    #     bbox_coords.append(evt.index)
    # elif brush_label in ['Positive (green)', 'Negative (red)']:
    #     click_coords.append(evt.index)
    #     click_labels.append(1 if brush_label == 'Positive (green)' else 0)
    # else:
    #     raise TypeError("Invalid brush label: {brush_label}")
    click_coords.append(current_coords)
    click_labels.append(current_label)

    # Only make new prediction if not waiting for additional bounding box click
    if (len(bbox_coords) % 2 == 0) and autopredict_checkbox:

        click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask = refresh_predictions(
            predictor, input_img, click_coords, click_labels, bbox_coords, current_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask,
            low_res_mask, img_features, binary_checkbox, multimask_mode
        )

        return click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask

    else:
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        )
        scribble_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, None, binary_checkbox
        )
        # Don't update output image if waiting for additional bounding box click
        click_input_viz = {
            'image': click_input_viz,
            'mask': scribble_img.get("mask") if scribble_img.get("mask") is not None else np.zeros((RES, RES, 3),
                                                                                                   dtype=np.uint32)
        }
        scribble_input_viz = {
            'image': scribble_input_viz,
            'mask': scribble_img.get("mask") if scribble_img.get("mask") is not None else np.zeros((RES, RES, 3),
                                                                                                   dtype=np.uint32)
        }
        return (click_input_viz, scribble_input_viz, output_img, best_mask, low_res_mask, img_features, click_coords,
                click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask)


def undo_click(predictor, input_img, brush_label, bbox_label, best_mask, low_res_mask, click_coords, click_labels,
               bbox_coords,
               seperate_scribble_masks, last_scribble_mask, scribble_img, img_features,
               output_img, binary_checkbox, multimask_mode, autopredict_checkbox):
    """
    Remove last click and then update the prediction
    """
    if bbox_label:
        if len(bbox_coords) > 0:
            bbox_coords.pop()
    elif brush_label in ['Positive (green)', 'Negative (red)']:
        if len(click_coords) > 0:
            click_coords.pop()
            click_labels.pop()
    else:
        raise TypeError("Invalid brush label: {brush_label}")

    # Only make new prediction if not waiting for additional bounding box click
    if (len(bbox_coords) == 0 or len(bbox_coords) == 2) and autopredict_checkbox:

        click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, seperate_scribble_masks, last_scribble_mask = refresh_predictions(
            predictor, input_img, click_coords, click_labels, bbox_coords, brush_label,
            scribble_img, seperate_scribble_masks, last_scribble_mask,
            low_res_mask, img_features, binary_checkbox, multimask_mode
        )
        return click_input_viz, scribble_input_viz, output_viz, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask

    else:
        click_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, seperate_scribble_masks, binary_checkbox
        )
        scribble_input_viz = viz_pred_mask(
            input_img, best_mask, click_coords, click_labels, bbox_coords, None, binary_checkbox
        )

        # Don't update output image if waiting for additional bounding box click
        return click_input_viz, scribble_input_viz, output_img, best_mask, low_res_mask, img_features, click_coords, click_labels, bbox_coords, seperate_scribble_masks, last_scribble_mask


def visualize_binary_mask(mask, title="Binary Mask Visualization"):
    """
    可视化二值掩码，支持形状为 (512, 512) 或 (2, 512, 512)，值应为 0 或 1。
    - 单通道：显示为白色前景
    - 双通道：channel 0 为绿色，channel 1 为红色（例如正/负标注）
    """
    plt.figure(figsize=(6, 6))

    if mask.ndim == 2:
        # 单通道，显示为灰度图（前景为白色）
        plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{title} (1-channel)")

    elif mask.ndim == 3 and mask.shape[0] == 2:
        # 双通道，合成为 RGB 可视化（绿色 + 红色）
        h, w = mask.shape[1:]
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)

        vis_img[mask[0] == 1] = [0, 255, 0]  # channel 0: green
        vis_img[mask[1] == 1] = [255, 0, 0]  # channel 1: red

        plt.imshow(vis_img)
        plt.title(f"{title} (2-channel: green/red)")

    else:
        raise ValueError(f"Unsupported shape: {mask.shape}. Expect (512, 512) or (2, 512, 512)")

    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{title}.png")


import numpy as np
import matplotlib.pyplot as plt


def visualize_rgb_image_and_mask(image, mask, title="Image and RGB Mask"):
    """
    可视化 RGB 图像和 RGB 掩码（不叠加），支持 shape = (512, 512, 3)
    
    参数:
    - image: ndarray, shape (H, W, 3)，原始图像
    - mask: ndarray, shape (H, W, 3)，彩色掩码图（可为伪彩色预测、三通道标签等）
    """
    assert image.ndim == 3 and image.shape[2] == 3, "image 必须是 (H, W, 3)"
    assert mask.ndim == 3 and mask.shape == image.shape, "mask 必须与 image 同形状"

    # 若图像是整数类型（0~255），转换为 float32 显示
    if image.dtype != np.float32 and image.max() > 1:
        image = image.astype(np.float32) / 255.0
    if mask.dtype != np.float32 and mask.max() > 1:
        mask = mask.astype(np.float32) / 255.0

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("RGB Mask")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{title}.png")


if __name__ == "__main__":
    pass
