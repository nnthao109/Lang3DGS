#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import torch.nn as nn

from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from autoencoder.model import Autoencoder
from preprocess import OpenCLIPNetworkConfig, OpenCLIPNetwork
from torch.utils.data import DataLoader
import cv2 
from sklearn.decomposition import PCA
import time

from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.decomposition import TruncatedSVD
# import cupy as cp  # Make sure you have CuPy installed
# from cuml import PCA as cumlPCA  # Make sure you have RAPIDS cuML installed

# def render_feature(feature):
#     breakpoint()
#     start_time_total = time.time()

#     # Step 1: Permute and reshape
#     start_time = time.time()
#     tensor_reshaped = feature.permute(1, 2, 0).reshape(-1, 32)
#     end_time = time.time()
#     print(f"Step 1 (Permute and reshape): {end_time - start_time:.4f} seconds")

#     # Step 2: Move to CPU and convert to NumPy
#     start_time = time.time()   
#     tensor_np = tensor_reshaped.cpu().numpy()
#     end_time = time.time()
#     print(f"Step 2 (CPU transfer and NumPy conversion): {end_time - start_time:.4f} seconds")

#     # Step 3: Apply PCA (various methods)
#     start_time = time.time()
    
#     if method == 'incremental':
#         pca = IncrementalPCA(n_components=3, batch_size=1000)
#         tensor_pca = pca.fit_transform(tensor_np)
#     elif method == 'randomized':
#         pca = PCA(n_components=3, svd_solver='randomized', random_state=42)
#         tensor_pca = pca.fit_transform(tensor_np)
#     elif method == 'truncated_svd':
#         svd = TruncatedSVD(n_components=3, random_state=42)
#         tensor_pca = svd.fit_transform(tensor_np)
#     elif method == 'gpu':
#         tensor_gpu = cp.asarray(tensor_np)
#         pca = cumlPCA(n_components=3)
#         tensor_pca = pca.fit_transform(tensor_gpu)
#         tensor_pca = cp.asnumpy(tensor_pca)
#     else:
#         pca = PCA(n_components=3)
#         tensor_pca = pca.fit_transform(tensor_np)

#     end_time = time.time()
#     print(f"Step 3 (PCA - {method}): {end_time - start_time:.4f} seconds")

#     # Step 4: Reshape the result back to image dimensions
#     start_time = time.time()
#     image_pca = tensor_pca.reshape(876, 1332, 3)
#     end_time = time.time()
#     print(f"Step 4 (Reshape to image dimensions): {end_time - start_time:.4f} seconds")

#     # Step 5: Normalize the image
#     start_time = time.time()
#     image_pca = (image_pca - image_pca.min()) / (image_pca.max() - image_pca.min())
#     end_time = time.time()
#     print(f"Step 5 (Normalize): {end_time - start_time:.4f} seconds")

#     # Step 6: Convert back to torch tensor and permute
#     start_time = time.time()
#     image_tensor = torch.tensor(image_pca).permute(2, 0, 1)
#     end_time = time.time()
#     print(f"Step 6 (Convert to tensor and permute): {end_time - start_time:.4f} seconds")

#     end_time_total = time.time()
#     print(f"Total time: {end_time_total - start_time_total:.4f} seconds")

#     return image_tensor

def render_feature_optimized(feature, method='linear_projection'):
    start_time_total = time.time()

    # Steps 1 and 2: Permute, reshape, and convert to NumPy
    tensor_np = feature.permute(1, 2, 0).reshape(-1, 32).cpu().numpy()

    # Step 3: Apply PCA (various methods)
    start_time = time.time()
    
    if method == 'incremental':
        pca = IncrementalPCA(n_components=3, batch_size=1000)
        tensor_pca = pca.fit_transform(tensor_np)
    elif method == 'randomized':
        pca = PCA(n_components=3, svd_solver='randomized', random_state=42)
        tensor_pca = pca.fit_transform(tensor_np)
    elif method == 'truncated_svd':
        svd = TruncatedSVD(n_components=3, random_state=42)
        tensor_pca = svd.fit_transform(tensor_np)
    elif method == 'gpu':
        tensor_gpu = cp.asarray(tensor_np)
        pca = cumlPCA(n_components=3)
        tensor_pca = pca.fit_transform(tensor_gpu)
        tensor_pca = cp.asnumpy(tensor_pca)
    else:
        pca = PCA(n_components=3)
        tensor_pca = pca.fit_transform(tensor_np)

    end_time = time.time()
    print(f"Step 3 (PCA - {method}): {end_time - start_time:.4f} seconds")

    # Remaining steps (4-6) are the same as before
    # ...

    end_time_total = time.time()
    print(f"Total time: {end_time_total - start_time_total:.4f} seconds")

    return tensor_pca

def render_feature_fast(feature, method='channel_selection', model=None, linear_projection_matrix=None):
    assert feature.shape == (32, 876, 1332), "Expected input shape [32, 876, 1332]"
    
    if method == 'linear_projection':
        # breakpoint()
        projection = linear_projection_matrix.to(feature.device)
        image = torch.matmul(projection, feature.view(32, -1)).view(3, 876, 1332)
    
    elif method == 'channel_selection':
        image = feature[[3, 15, 30]]
    
    elif method == 'average_pooling':
        # Reshape to group channels, then average
        grouped = feature.view(3, 10, 876, 1332)  # Group into 3 sets of 10 channels (the last group will have 12)
        image = grouped.mean(dim=1)
    
    elif method == 'random_projection':
        random_matrix = torch.randn(3, 32).to(feature.device)
        image = torch.matmul(random_matrix, feature.view(32, -1)).view(3, 876, 1332)
    
    elif method == 'argmax':
        # Take the argmax along the first dimension to get the most likely class
        segmentation_map = torch.argmax(feature, dim=0)
        
        # Create a colormap
        num_classes = feature.shape[0]
        import matplotlib.pyplot as plt
        colormap = plt.get_cmap('jet')
        
        # Apply colormap to segmentation map
        colored_segmentation = colormap(segmentation_map.float() / (num_classes - 1))
        
        # Convert to torch tensor with shape [3, 876, 1332]
        image = torch.from_numpy(colored_segmentation[:, :, :3].transpose(2, 0, 1)).to(feature.device)
    
    elif method == 'conv':
        # breakpoint()
        model = model.to(feature.device)
        reduced_feature = model(feature.unsqueeze(0)).squeeze(0)
        return reduced_feature
    
    elif method == 'pca_lowrank':
        # Move feature to CPU and permute to (876 * 1332, 32) shape
        tensor_np = feature.permute(1, 2, 0).reshape(-1, 32).cpu()

        # Perform PCA using torch.pca_lowrank (k=3 for reducing to 3 components)
        U, S, V = torch.pca_lowrank(tensor_np, q=3)

        # Project the data onto the first 3 principal components
        tensor_pca = torch.matmul(tensor_np, V[:, :3])

        # Reshape the result back to image dimensions (876, 1332, 3)
        image_pca = tensor_pca.reshape(876, 1332, 3)

        # Normalize the image for saving (optional, based on your data range)
        image_pca = (image_pca - image_pca.min()) / (image_pca.max() - image_pca.min())

        # Convert back to torch tensor and permute to (3, 876, 1332)
        image_tensor = torch.tensor(image_pca).permute(2, 0, 1)  # (3, 876, 1332)

        return image_tensor
    
    else:
        raise ValueError("Unknown method")
    return image

# Compute PCA for the first frame and return the principal components (V)
def compute_pca_for_first_frame(feature_):
    """feature on GPU"""
    tensor_flat = feature_.permute(1, 2, 0).reshape(-1, 32)  # Reshape to (876 * 1332, 32)
    
    # Perform PCA (low-rank approximation) to reduce to 3 principal components
    U, S, V = torch.pca_lowrank(tensor_flat, q=3)
    
    return V  # V will be reused for subsequent frames

# Apply the saved PCA components (V) to a new feature map (reuse PCA)
def render_pca_lowrank_based_on_firstframe(feature, V_first_frame=None):
    
    start_time_1 = time.time()
    tensor_flat = feature.permute(1, 2, 0).reshape(-1, 32)  # Reshape to (876 * 1332, 32)
    end_time_2 = time.time()
    print(f"Reshape time: {end_time_2 - start_time_1:.4f} seconds")
    
    # Project the data onto the first 3 principal components using saved V
    start_time_matmul = time.time()
    tensor_pca = torch.matmul(tensor_flat, V_first_frame[:, :3])
    end_time_matmul = time.time()
    print(f"Matrix multiplication time: {end_time_matmul - start_time_matmul:.4f} seconds")
    
    # Reshape back to the image dimensions (876, 1332, 3)
    image_pca = tensor_pca.reshape(720, 1280, 3)
    
    # Normalize the image (optional, depends on your data)
    image_pca = (image_pca - image_pca.min()) / (image_pca.max() - image_pca.min())
    
    # Convert back to a torch tensor and permute to (3, 876, 1332)
    image_tensor = torch.tensor(image_pca).permute(2, 0, 1).contiguous()  # Shape (3, 876, 1332)
    
    return image_tensor

"""
GPU Based PCA low rank
"""


def render_feature_pca(feature):
    # Move feature to CPU and permute to (876 * 1332, 32) shape
    tensor_np = feature.permute(1, 2, 0).reshape(-1, 32).cpu()

    # Perform PCA using torch.pca_lowrank (k=3 for reducing to 3 components)
    U, S, V = torch.pca_lowrank(tensor_np, q=3)

    # Project the data onto the first 3 principal components
    tensor_pca = torch.matmul(tensor_np, V[:, :3])

    # Reshape the result back to image dimensions (876, 1332, 3)
    image_pca = tensor_pca.reshape(876, 1332, 3)

    # Normalize the image for saving (optional, based on your data range)
    image_pca = (image_pca - image_pca.min()) / (image_pca.max() - image_pca.min())

    # Convert back to torch tensor and permute to (3, 876, 1332)
    image_tensor = torch.tensor(image_pca).permute(2, 0, 1)  # (3, 876, 1332)

    return image_tensor

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_npy")
    gts_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_npy")

    makedirs(render_npy_path, exist_ok=True)
    makedirs(gts_npy_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        output = render(view, gaussians, pipeline, background, args)

        if not args.include_feature:
            rendering = output["render"]
        else:
            rendering = output["language_feature_image"]
            rendering = render_feature(rendering)
            
        if not args.include_feature:
            gt = view.original_image[0:3, :, :]
            
        else:
            gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)
            gt = render_feature(gt)
            
        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"),rendering.permute(1,2,0).cpu().numpy())
        np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"),gt.permute(1,2,0).cpu().numpy())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

# def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args, gaussians):
    
def calculate_selection_score(features, query_features, score_threshold=0.178, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        if scores.shape[-1] == 1:
            scores = scores[:, 0]  # (N_points,)
            scores = (scores >= score_threshold).float()
        else:
            scores = torch.nn.functional.softmax(scores, dim=-1)  # (N_points, n_texts)
            if score_threshold is not None:
                scores = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = (scores >= score_threshold).float()
            else:
                scores[:, positive_ids[0]] = scores[:, positive_ids].sum(-1)  # (N_points, )
                scores = torch.isin(torch.argmax(scores, dim=-1), torch.tensor(positive_ids).cuda()).float()
        return scores

def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        print("score:", scores)
        if scores.shape[-1] == 1:
            scores = scores[:, 0]  # (N_points,)
            mask = (scores >= score_threshold)
        
        return mask
if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset = model.extract(args)
    safe_state(args.quiet)
    #load Gaussian
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #load Clip
    config = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16",
        clip_model_pretrained="laion2b_s34b_b88k",
        clip_n_dims=512,
        positives=("",),
        negatives=("",)
    )
    model_clip = OpenCLIPNetwork(config)
    model_clip = model_clip.to("cuda")
    #load Autoencoder 
    ckpt_path = "/home/thaonn/LangSplat/autoencoder/ckpt/drjohnson_32/best_ckpt.pth"
    checkpoint = torch.load(ckpt_path)
    encoder_hidden_dims = [256, 128, 64, 32]
    decoder_hidden_dims = [64, 128, 256, 512]
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    model.load_state_dict(checkpoint)
    model.eval()
    # 
    
    print(gaussians.get_language_feature.shape)
    latent_features = gaussians.get_language_feature
    with torch.no_grad():  # Disable gradients for inference
        features = model.decode(latent_features)
    
    print(features.shape)
    
    # Mapping language and features
    # image_path = '/home/thaonn/LangSplat/dog.jpeg'
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = torch.from_numpy(image)
    # image = image.float() / 255.0
    # image = image.permute(2, 0, 1).to('cuda')
    # image_embed = model_clip.encode_image(image.unsqueeze(0))
    textures = model_clip.extract_text_feature(["chair",])
    model_clip.set_positives(["chair",])
    # print(textures.shape)
    textures = textures.to(torch.float32)
    features = features.to(torch.float32)
    # print("score ",calculate_selection_score(features, textures))
    relevancy_map = model_clip.get_relevancy(features,0)
    mask2 = calculate_selection_score_delete(features,textures, 0.25 )
    # print("relevancy_map", relevancy_map[:])
    # threshold = 0.475
    # mask = relevancy_map[:, 0] > threshold
    print("mask", mask2)
    
    indices = torch.nonzero(mask2, as_tuple=False)

# # Print the indices
    print(indices.shape)
    mask = mask2.squeeze()
    # print("gaussian_mask",gaussians._opacity[mask] )
    # gaussians._opacity = gaussians._opacity.clone()
    # gaussians._opacity[mask] = -999
    device = gaussians._features_dc.device  # Get the device of _features_dc
    new_color_value = torch.tensor([1.0, 0.0, 0.0])
    
    # Move new_color_value to the same device
    new_color_value = new_color_value.to(device)

    # Apply the color change to the masked Gaussians
    gaussians._features_dc = gaussians._features_dc.clone()
    gaussians._features_dc[mask] = new_color_value
    
    # gaussians._opacity[mask] = -999
#     # print(model_clip.get_relevancy(features,0).shape)
#     print(gaussians.get_opacity[134])
#     opacity_zero_mask = gaussians.get_opacity == 0.

#     # Get the indices where the opacity is 0
#     zero_indices = torch.nonzero(opacity_zero_mask, as_tuple=False)

#     # Print the indices where opacity is 0
#     print(zero_indices)
#     # sim = torch.einsum("cq,dq->dc", textures, features)
#     # label_soft = sim.softmax(dim=1)
#     # label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
#     # print(label_soft)
    with torch.no_grad():
        render_set(dataset.model_path, dataset.source_path, "train_sem_32_full", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

