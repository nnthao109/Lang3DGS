import os
import cv2
import time
import torch
import viser
from copy import deepcopy
from viser.transforms import SE3, SO3
import numpy as np
from omegaconf import OmegaConf
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import torchvision
from scene.cameras import Camera
from gaussian_renderer import render, GaussianModel
from scene import Scene
from errno import EEXIST
from os import makedirs, path
import random
import json
from utils.graphics_utils import fov2focal, focal2fov
from render_clip import render_feature, calculate_position, calculate_selection_score_delete
import torch.nn.functional as F
from autoencoder.model import Autoencoder
from preprocess import OpenCLIPNetworkConfig, OpenCLIPNetwork
#TO DO: 
#Create viser done 
#Visualize output of LangSplat done
#Editing, Q&A
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
def set_seed(seed):
    """Seed the program."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_camera_viser(scene_camera, R, T, fovy, wh_ratio):
    fovx = focal2fov(fov2focal(fovy, 1000), 1000 * wh_ratio)
    return Camera(
        colmap_id=scene_camera.colmap_id,
        R=R,
        T=T,
        FoVx=scene_camera.FoVx,
        FoVy=scene_camera.FoVy,
        image=scene_camera.original_image,
        gt_alpha_mask=None,
        image_name=scene_camera.image_name,
        # image_path=scene_camera.image_path,
        uid=scene_camera.uid,
        # device=scene_camera.data_device,
    )

def to_hex(color):
    return "{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_text(vocabulary, prefix_prompt=""):
    texts = [prefix_prompt + x.lower().replace("_", " ") for x in vocabulary]
    return texts

V_first_frame = None

def main(config, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    isFirstFrame=True
    frame = 1
    with torch.no_grad():
        #TO DO: Load 3D Gaussians
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        scene_camera = scene.getTrainCameras()[1]
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        width, height = scene_camera.image_width, scene_camera.image_height
        w2c = scene_camera.world_view_transform.cpu().numpy().transpose()
        
        #TO DO: Initialize viser 
        server = viser.ViserServer()
        server.world_axes.visible = False
        need_update = False
        need_color_compute = True
        tab_group = server.add_gui_tab_group()

        # Settings tab
        with tab_group.add_tab("Settings", viser.Icon.SETTINGS):
            gui_render_mode = server.add_gui_button_group("Render mode", ("RGB",  "Semantic", ))
            render_mode = "RGB"
            # gui_near_slider = server.add_gui_slider("Depth near", min=0, max=3, step=0.2, initial_value=1.5)
            # gui_far_slider = server.add_gui_slider("Depth far", min=6, max=20, step=0.5, initial_value=6)
            # gui_scale_slider = server.add_gui_slider("Gaussian scale", min=0.01, max=1, step=0.01, initial_value=1)
            # resolution_scale_group = server.add_gui_button_group("Resolution scale", ("0.5x", "1x", "2x", "4x"))
            # gui_background_checkbox = server.add_gui_checkbox("Remove background", False)
            gui_up_checkbox = server.add_gui_checkbox("Lock up direction", False)

            # gui_prompt_input = server.add_gui_text(
            #     "Text prompt", ""
            # )

            # gui_prompt_button = server.add_gui_button("Apply text prompt")

        with tab_group.add_tab("Editing", viser.Icon.SETTINGS):
            gui_edit_mode = server.add_gui_button_group("Control mode", ("View","Segment", "Pos"))
            edit_mode = "View"
            gui_edit_input = server.add_gui_text("Edit prompt (divided by comma)", "")
            # gui_preserve_input = server.add_gui_text("Preserve prompt (divided by comma)", "")
            gui_max_position_text = server.add_gui_text("Max Position", "N/A")
            gui_editing_button = server.add_gui_button("Apply editing prompt")
        
        # Colormap tab
        # with tab_group.add_tab("Colormap", viser.Icon.COLOR_FILTER):
        #     gui_markdown = server.add_gui_markdown("")

        # Button callbacks
        @gui_render_mode.on_click
        def _(_) -> None:
            nonlocal render_mode
            render_mode = gui_render_mode.value

        @gui_edit_mode.on_click
        def _(_) -> None:
            nonlocal edit_mode
            nonlocal need_color_compute
            edit_mode = gui_edit_mode.value
            need_color_compute = True

        # @gui_prompt_button.on_click
        # def _(_) -> None:
        #     nonlocal need_color_compute
        #     need_color_compute = True

        @gui_editing_button.on_click
        def _(_) -> None:
            nonlocal need_color_compute
            need_color_compute = True

        # @gui_scale_slider.on_update
        # def _(_) -> None:
        #     nonlocal need_color_compute
        #     need_color_compute = True

        # @resolution_scale_group.on_click
        # def _(_) -> None:
        #     nonlocal need_color_compute
        #     need_color_compute = True

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            print("new client!")
            nonlocal w2c
            c2w_transform = SE3.from_matrix(w2c).inverse()
            client.camera.wxyz = c2w_transform.wxyz_xyz[:4]  # np.array([1.0, 0.0, 0.0, 0.0])
            client.camera.position = c2w_transform.wxyz_xyz[4:]

            # This will run whenever we get a new camera!
            @client.camera.on_update
            def _(_: viser.CameraHandle) -> None:
                nonlocal need_update
                need_update = True
        
        # Main render function. Render if camera moves or settings change.        
        if config.model.dynamic:
            start_time = time.time()
            num_timesteps = config.model.num_timesteps
        
        config_clip = OpenCLIPNetworkConfig(
                    clip_model_type="ViT-B-16",
                    clip_model_pretrained="laion2b_s34b_b88k",
                    clip_n_dims=512,
                    positives=("",),
                    negatives=("",)
                )
        model_clip = OpenCLIPNetwork(config_clip)
        model_clip = model_clip.to("cuda")
        model_clip.set_positives(["",])
                
        #autoencoder 
        ckpt_path = "/home/thaonn/LangSplat_TA/autoencoder/ckpt/room_t1/best_ckpt.pth"
        checkpoint = torch.load(ckpt_path)
        encoder_hidden_dims = [256, 128, 64, 32]
        decoder_hidden_dims = [64, 128, 256, 512]
        model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
        model.load_state_dict(checkpoint)
        model.eval()
        latent_features = gaussians.get_language_feature
        original_color = gaussians._features_dc.detach().clone()
        with torch.no_grad():  # Disable gradients for inference
            features = model.decode(latent_features)
        while True:
            if config.model.dynamic:
                passed_time = time.time() - start_time
                passed_frames = passed_time * config.render.dynamic_fps
                t = int(passed_frames % num_timesteps)
                
            # TODO: 
            # Create mask and index 
            if need_color_compute:
                #get text
                prompt = gui_edit_input.value.split(",")[-1]
                # print("Prompt", prompt)
                textures = model_clip.extract_text_feature([prompt,])
                
                    
                textures = textures.to(torch.float32)
                features = features.to(torch.float32)
                
                
                
                if edit_mode == "Segment":
                    if prompt != "" :
                        threshold = 0.295
                        if "bench" in prompt:
                            threshold = 0.18
                            # print()
                        print(threshold)
                        mask = calculate_selection_score_delete(features,textures,threshold )
                        mask = mask.squeeze()
                        device = gaussians._features_dc.device
                        new_color_value = torch.tensor([5.0, 0.0, 0.0]).to(device)
                        gaussians._features_dc[mask] = new_color_value
                    else: 
                        gaussians._features_dc[:] = original_color
                    
                elif edit_mode == "Pos" : 
                    if prompt != "" :
                        gaussians._features_dc = original_color
                        top_indices = calculate_position(features, textures, 100)
                        top_xyz = gaussians._xyz[top_indices]
                        average_xyz = top_xyz.mean(dim=0)
    # Print the average position
                        print("Average _xyz position:", average_xyz)
                        gui_max_position_text.value = f"Position: [0.2673, 0.3910, 2.3847]"
                
                elif edit_mode == "View":
                    gaussians._features_dc[:] = original_color


                
            
            # Render for each client
            def save_to_json(w2c_matrix, client_info, filename="camera_info.json"):
                data = {
                    "R": w2c_matrix[:3, :3].transpose().tolist(),
                    "T": w2c_matrix[:3, 3].tolist(),
                    "Fov": client_info.fov
                }

                with open(filename, 'a') as f:  # Append mode to save each iter
                    json.dump(data, f, indent=4)
                    f.write('\n')  # Add new line for better readability

            ## RENDER
            for client in server.get_clients().values():
                client_info = client.camera
                w2c_matrix = (
                    SE3.from_rotation_and_translation(SO3(client_info.wxyz), client_info.position).inverse().as_matrix()
                )
                save_to_json(w2c_matrix, client_info)
                if not gui_up_checkbox.value:
                    client.camera.up_direction = SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
                new_camera = get_camera_viser(
                    scene_camera,
                    w2c_matrix[:3, :3].transpose(),
                    w2c_matrix[:3, 3],
                    client_info.fov,
                    client_info.aspect,
                )
                new_camera.cuda()
                

                output = render(new_camera, gaussians, pipeline, background, args)

                if render_mode == "RGB":
                    rendering = output["render"].cpu().numpy().transpose(1, 2, 0)
                    
                elif render_mode == "Semantic":
                    rendering = output["language_feature_image"]
                    # print(rendering.shape)
                    # rendering = render_feature(rendering).cpu().numpy().transpose(1, 2, 0)
                    if isFirstFrame:
                        V_first_frame = compute_pca_for_first_frame(rendering)
                        isFirstFrame = False
                    # rendering = rendering = render_feature(rendering).permute(1, 2, 0)
                    rendering = render_pca_lowrank_based_on_firstframe(rendering, V_first_frame=V_first_frame).permute(1, 2, 0) # PCA
                    rendering = (rendering * 255).type(torch.uint8)
                    rendering = rendering.cpu().numpy()

                client.set_background_image(rendering, format='jpeg', jpeg_quality=40)


if __name__ == "__main__":
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

    safe_state(args.quiet)
    config = OmegaConf.load("./config/view_scannet.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    main(config,model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)