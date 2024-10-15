import torch
import numpy as np

# Replace 'language_feature_name' with the actual name you are using
# language_feature_name = 'your_file_name_here'  # e.g., 'data/file'

# Load the numpy array from the .npy file and convert it into a PyTorch tensor
seg_map = torch.from_numpy(np.load("/home/thaonn/LangSplat/42081_shutdown/language_features/42081_0_gt_s.npy"))
# print(seg_map)
# Print the size (shape) of the tensor
print("Shape of seg_map:", seg_map.shape)  # or seg_map.size()