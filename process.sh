#!/bin/bash

# get the language feature of the scene
python preprocess.py --dataset_name $dataset_path

# train the autoencoder
cd autoencoder
python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name ae_ckpt
# e.g. python train.py --dataset_path /home/thaonn/LangSplat/preprocessed_dataset/sofa --encoder_dims 256 12 64 32 168 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa
#      python train.py --dataset_path /home/thaonn/gaussian-splatting/sample_data/db/drjohnson --encoder_dims 256 128 --decoder_dims 256 512 --lr 0.0007 --dataset_name drjohnson_128


# get the 3-dims language feature of the scene
python test.py --dataset_name $dataset_path --dataset_name $dataset_name
# e.g. python test.py --dataset_path ../data/sofa --dataset_name sofa
#      python test.py --dataset_path /home/thaonn/gaussian-splatting/sample_data/db/drjohnson --encoder_dims 256 --decoder_dims 512 --dataset_name drjohnson

# ATTENTION: Before you train the LangSplat, please follow https://github.com/graphdeco-inria/gaussian-splatting
# to train the RGB 3D Gaussian Splatting model.
# put the path of your RGB model after '--start_checkpoint'

for level in 1 2 3
do
    python train.py -s $dataset_path -m output/${casename} --start_checkpoint $dataset_path/$casename/chkpnt30000.pth --feature_level ${level}
    # e.g. python train.py -s /home/thaonn/LangSplat/preprocessed_dataset/sofa -m /home/thaonn/LangSplat/output/sofa_2 --start_checkpoint /home/thaonn/LangSplat/output/sofa_2/chkpnt30000.pth --feature_level 3
    # python train.py -s /home/thaonn/gaussian-splatting/sample_data/db/drjohnson -m /home/thaonn/gaussian-splatting/output/bf6b8c51-f --start_checkpoint /home/thaonn/gaussian-splatting/output/bf6b8c51-f/chkpnt30000.pth --feature_level 3
    #python train.py -s /home/thaonn/LangSplat/preprocessed_dataset/sofa -m /home/thaonn/LangSplat/preprocessed_dataset/sofa/sofa --start_checkpoint /home/thaonn/LangSplat/preprocessed_dataset/sofa/sofa/chkpnt30000.pth --feature_level 3
done

for level in 1 2 3
do
    # render rgb
    python render.py -m output/${casename}_${level}
    # render language features
    python render.py -m output/${casename}_${level} --include_feature
    # e.g. python render.py -m output/sofa_3 --include_feature
    # python render.py -m /home/thaonn/gaussian-splatting/output/bf6b8c51-f_3 --include_feature
    # python thinhvd_view_viser.py -m /home/thaonn/LangSplat_TA/output_raw_-1_3 --include_feature
    # python view_viser.py -m /home/thaonn/gaussian-splatting/output/bf6b8c51-f_3 --include_feature
    # python view_viser.py -m /home/thaonn/gaussian-splatting/output/playroom_3 --include_feature
done