import torch
import torchvision
from preprocess import OpenCLIPNetworkConfig, OpenCLIPNetwork
import cv2
import os
from gaussian_renderer import render, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from gaussian_renderer import render
from render import render_set
from utils.general_utils import safe_state
# from scene import Scene
#TODO 
#Load & test CLIP model 
#Load 3D Gaussian model then extract feature semantic
#Save images to check 
    #Save RGB
    #Save semantic 



    
config = OpenCLIPNetworkConfig(
    clip_model_type="ViT-B-16",
    clip_model_pretrained="laion2b_s34b_b88k",
    clip_n_dims=512,
    positives=("cat", "dog", "horse"),
    negatives=("car", "building", "chair")
)
#Load 3D GS 
sh_degree = 3
gaussians = GaussianModel(sh_degree)
model_path = "/home/thaonn/gaussian-splatting/output/bf6b8c51-f_3"
checkpoint = os.path.join(model_path, 'chkpnt30000.pth')
(model_params, first_iter) = torch.load(checkpoint)
parser = ArgumentParser(description="Testing script parameters")
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--include_feature", action="store_true")
args = get_combined_args(parser)
gaussians.restore(model_params, args, mode='test')
print(gaussians.get_language_feature.shape)
safe_state(args.quiet)
def render_semantic(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        scene = Scene(dataset, gaussians, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, dataset.source_path, "train_semantic", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
            render_set(dataset.model_path, dataset.source_path, "test_semantic", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)
render_semantic(model.extract(args),  args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)
# Instantiate the OpenCLIPNetwork with the defined config
model_clip = OpenCLIPNetwork(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU
model_clip = model_clip.to(device)
textures = model_clip.extract_text_feature(("cat", "dog", "horse"))
# print(textures.shape)

image_path = '/home/thaonn/LangSplat/dog.jpeg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image)
image = image.float() / 255.0
image = image.permute(2, 0, 1).to('cuda')
image_embed = model_clip.encode_image(image.unsqueeze(0))
# print(model.encode_image(image.unsqueeze(0)).shape)

sim = torch.einsum("cq,dq->dc", textures, image_embed)
label_soft = sim.softmax(dim=1)
label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
# print(label_hard)
positive_id = 1  # Assuming the first positive phrase "cat"
relevancy_scores = model_clip.get_relevancy(image_embed, positive_id)

# # Print relevancy scores
# print(relevancy_scores)

print("Rendering " + args.model_path)

