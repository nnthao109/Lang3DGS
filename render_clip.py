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
import torch
from sklearn.decomposition import PCA
import time

def render_feature(feature):
    # Move feature to CPU and permute to (876 * 1332, 32) shape
    torch.manual_seed(seed=42)
    tensor_np = feature.permute(1, 2, 0).reshape(-1, 32).cpu()

    # Perform PCA using torch.pca_lowrank (k=3 for reducing to 3 components)
    U, S, V = torch.pca_lowrank(tensor_np, q=3)

    # Project the data onto the first 3 principal components
    tensor_pca = torch.matmul(tensor_np, V[:, :3])

    # Reshape the result back to image dimensions (876, 1332, 3)
    image_pca = tensor_pca.reshape(720, 1280, 3)

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
            start_time = time.time()
            rendering = render_feature(rendering)
            end_time_total = time.time()
            print(f"Rendering time: {end_time_total - start_time_total:.4f} seconds")
            
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
# def calculate_position(features, query_features):
#     features /= features.norm(dim=-1, keepdim=True)
#     query_features /= query_features.norm(dim=-1, keepdim=True)
#     scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
#     scores = scores[:, 0]
#     max_indices = torch.argmax(scores)
#     return max_indices
def calculate_position(features, query_features, top = 100):
    features /= features.norm(dim=-1, keepdim=True)
    query_features /= query_features.norm(dim=-1, keepdim=True)
    scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
    scores = scores[:, 0]
    
    # Get the top 100 indices
    top_indices = torch.topk(scores, top).indices
    
    return top_indices

def calculate_selection_score_delete(features, query_features, score_threshold=None, positive_ids=[0]):
        features /= features.norm(dim=-1, keepdim=True)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        scores = features.half() @ query_features.T.half()  # (N_points, n_texts)
        # print("score:", scores)
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
    ckpt_path = "/home/thaonn/LangSplat_TA/autoencoder/ckpt/room_t1/best_ckpt.pth"
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
    
    
    ####################################### USING TEXT QUERY MODEL ############################
    textures = model_clip.extract_text_feature(["a brown minimalist, industrial-style leather bench",])
    model_clip.set_positives(["",])
    textures = textures.to(torch.float32)
    features = features.to(torch.float32)
    mask2 = calculate_selection_score_delete(features,textures, 0.2 )
    print("mask", mask2)
    indices = torch.nonzero(mask2, as_tuple=False)
    print(indices.shape)
    mask = mask2.squeeze()
    ####################################### REMOVE OBJECT ############################
    # gaussians._opacity = gaussians._opacity.clone()
    # gaussians._opacity[mask] = -999
    
    ####################################### CHANGE COLOR ############################
    device = gaussians._features_dc.device  # Get the device of _features_dc
    new_color_value = torch.tensor([5.0, 0.0, 0.0])
    new_back_color = torch.tensor([0.0,0.0,0.0])
    # Move new_color_value to the same device
    new_color_value = new_color_value.to(device)
    new_back_color = new_back_color.to(device)
    # # Apply the color change to the masked Gaussians
    gaussians._features_dc = gaussians._features_dc.clone()
    gaussians._features_rest = gaussians._features_rest.clone()
    gaussians._features_dc[mask] = new_color_value
    
    ####################################### GET POSITION ############################
    # index = calculate_position(features,textures)
    # print("index", index)
    # print("MAX position", gaussians._xyz[index])
    top_indices = calculate_position(features, textures, 100)
    top_xyz = gaussians._xyz[top_indices]
    average_xyz = top_xyz.mean(dim=0)
    # Print the average position
    print("Average _xyz position:", average_xyz)
    ####################################### RENDER IMAGE  ############################
    # with torch.no_grad():
    #     render_set(dataset.model_path, dataset.source_path, "train_sem_32_full", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

