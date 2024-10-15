import os
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torchvision.transforms as T
from dataclasses import dataclass, field
from typing import Tuple, Type
from torch import nn
import sys

# Add the FastSAM directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FastSAM'))

from FastSAM.fastsam import FastSAM, FastSAMPrompt
try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed. Install it with `pip install open_clip-torch`")

from FastSAM.fastsam import FastSAM, FastSAMPrompt

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = T.Compose(
            [
                T.Resize((224, 224)),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = self.tokenizer(self.positives).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = self.tokenizer(self.negatives).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return f"openclip_{self.config.clip_model_type}_{self.config.clip_model_pretrained}"

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = self.tokenizer(self.positives).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def create(image_list, data_list, save_folder, model, fast_sam_generator):
    assert image_list is not None, "image_list must be provided to generate features"
    img_embeds_list = []
    seg_maps_list = []

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", total=len(image_list)):
        try:
            img_embed, seg_map = embed_clip_sam(img, model, fast_sam_generator)
        except Exception as e:
            raise RuntimeError(f"Error processing image {i}: {e}")

        img_embeds_list.append(img_embed)
        seg_maps_list.append(seg_map)

    for i in range(len(img_embeds_list)):
        save_path = os.path.join(save_folder, os.path.splitext(data_list[i])[0])
        curr = {
            'feature': img_embeds_list[i],
            'seg_maps': seg_maps_list[i]
        }
        save_numpy(save_path, curr)


def save_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())
    
    
# Convert masks to boolean (True/False)
def masks_to_bool(masks):
    if type(masks) == np.ndarray:
        return masks.astype(bool)
    return masks.cpu().numpy().astype(bool)

def embed_clip_sam(image, clip_model, fast_sam_generator):
    image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = fast_sam_generator(source=image_rgb,
                               retina_masks=True,
                               conf = 0.5,
                               iou=0.6)
    prompt_process = FastSAMPrompt(image_rgb, results)
    masks = prompt_process.everything_prompt()
    # areas = masks.
    # breakpoint()
    masks_processed = masks_to_bool(masks.masks.data)
    # breakpoint()

    seg_map = -np.ones(image.shape[1:], dtype=np.int32)
    seg_img_list = []
    for idx in range(len(masks_processed)):
        seg_mask = masks_processed[idx]
        seg_map[seg_mask] = idx
        # breakpoint()
        # Extract the masked region
        x, y, w, h = masks.boxes.data[idx][:4].int()
        # x, y, w, h = map(int, data[0, :4].tolist())
        seg_img = image_rgb.copy()
        seg_img[~seg_mask] = 0
        seg_img = seg_img[y:y+h, x:x+w]

        # Pad and resize to 224x224
        seg_img = pad_and_resize(seg_img, (224, 224))
        seg_img_list.append(seg_img)

    if not seg_img_list:
        raise RuntimeError("No valid segment images extracted.")

    # Stack and process segmentation images
    seg_imgs = np.stack(seg_img_list, axis=0)  # N x H x W x 3
    seg_imgs = torch.from_numpy(seg_imgs.astype(np.float32)).permute(0, 3, 1, 2) / 255.0
    seg_imgs = seg_imgs.to('cuda')
    with torch.no_grad():
        embeddings = clip_model.encode_image(seg_imgs.half())
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings.cpu().half()

    return embeddings, torch.from_numpy(seg_map)


def pad_and_resize(img, size):
    h, w, _ = img.shape
    max_side = max(h, w)
    pad_img = np.zeros((max_side, max_side, 3), dtype=img.dtype)
    if h > w:
        x_offset = (h - w) // 2
        pad_img[:, x_offset:x_offset + w, :] = img
    else:
        y_offset = (w - h) // 2
        pad_img[y_offset:y_offset + h, :, :] = img
    resized_img = cv2.resize(pad_img, size)
    return resized_img


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True


def main():
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/FastSAM.pt")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    # Initialize models
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig())
    # sam_model = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam_model,
    #     points_per_side=32,
    #     pred_iou_thresh=0.7,
    #     stability_score_thresh=0.85,
    #     min_mask_region_area=100,
    # )
    fast_sam = FastSAM(sam_ckpt_path)

    # Load and preprocess images
    img_list = []
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Failed to read image {image_path}. Skipping.")
            continue

        orig_h, orig_w = image.shape[:2]
        if args.resolution != -1:
            scale = orig_w / args.resolution
            resolution = (args.resolution, int(orig_h / scale))
            image = cv2.resize(image, resolution)
        elif orig_h > 1080:
            scale = orig_h / 1080
            resolution = (int(orig_w / scale), 1080)
            image = cv2.resize(image, resolution)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        img_list.append(image_tensor)

    images = img_list
    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)

    create(images, data_list, save_folder, clip_model, fast_sam)


if __name__ == '__main__':
    main()
