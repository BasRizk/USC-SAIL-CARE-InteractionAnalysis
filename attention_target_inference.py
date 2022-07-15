import os
import pickle
import argparse
import numpy as np
import plotly.express as px
import pandas as pd
from PIL import Image
from torchvision import transforms


import sys
REPO_PATH = '/proj/brizk/attention-target-detection'
sys.path.append(REPO_PATH)
from config import input_resolution, output_resolution
from utils import imutils, evaluation
from model import ModelSpatial
import torch

import cv2
from tqdm import tqdm
from dataset_loader import DatasetLoader
from annotations_loader import RetinafaceInferenceGenerator


parser = argparse.ArgumentParser(description='Attention-Target')
parser.add_argument('--home_dir', default='/proj/brizk/output', type=str, help='home directory')
parser.add_argument('--save_folder', default='attentiontarget', type=str, help='prefixed by home_dir, dir to save results')
args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)
   
def setup_model():
    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(REPO_PATH, 'model_demo.pt'))
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)
    return model

if __name__ == "__main__":

    dataset = DatasetLoader()
    faces_dir = os.path.join(args.home_dir, 'retinaface')
    faces_files = RetinafaceInferenceGenerator(faces_dir)
    column_names = ['frame', 'confidence', 'left', 'top', 'right', 'bottom']
    target_dir = os.path.join(*[args.home_dir, args.save_folder])
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    logger_file = open(os.path.join(target_dir,'recent_run_log.txt'), "w")

    model = setup_model()
    # set up data transformation
    test_transforms = _get_transform()
    
    with torch.no_grad():
        for ds, filename in tqdm(faces_files):
            video_name = filename.split(".")[0]
            print('Starting video ' + ds + '/' + video_name)

            logger_file.write('Starting video ' + ds + '/' + video_name + '\n')
            logger_file.flush()

            ds_dir = os.path.join(target_dir, ds)
            if not os.path.exists(ds_dir):
                os.makedirs(ds_dir)
                

            inference_filepath = os.path.join(*[ds_dir, video_name + ".pkl"])
            if os.path.exists(inference_filepath):
                print('Skipping ' + filename + ' as already exists.')                
                logger_file.write('Skipping' + filename + 'as already exists\n')
                logger_file.flush()
                continue
            inference_data = []
            
            df = pd.read_csv(
                os.path.join(*[faces_dir, ds, filename]),
                header=None, names=column_names, usecols=range(6)
            )

        
            df['left'] -= (df['right']-df['left'])*0.1
            df['right'] += (df['right']-df['left'])*0.1
            df['top'] -= (df['bottom']-df['top'])*0.1
            df['bottom'] += (df['bottom']-df['top'])*0.1

            video = dataset[(ds, video_name)]
            for i in tqdm(df.index):
                frame_num = df.loc[i, 'frame']
                frame_raw = Image.fromarray(
                    cv2.cvtColor(video[frame_num], cv2.COLOR_BGR2RGB)
                )
                # frame_raw = Image.open(os.path.join(*[args.home_dir, 'imgs', 'MS' , video_name, str(frame_num) + ".jpg" ]))
                width, height = frame_raw.size

                head_box = np.array([df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]).astype(np.int32)
                head_img = frame_raw.crop((head_box)) # head crop
                head = test_transforms(head_img) # transform inputs
        
                frame = test_transforms(frame_raw)
                
                head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                            resolution=input_resolution).unsqueeze(0)

                head = head.unsqueeze(0).cuda()
                frame = frame.unsqueeze(0).cuda()
                head_channel = head_channel.unsqueeze(0).cuda()
                
                # forward pass
                raw_hm, _, inout = model(frame, head_channel, head)

                # # heatmap modulation
                raw_hm = raw_hm.cpu().detach().numpy() * 255
                raw_hm = raw_hm.squeeze()
                inout = inout.cpu().detach().numpy()
                inout = 1 / (1 + np.exp(-inout))
                inout = (1 - inout) * 255
                # norm_map = imresize(raw_hm, (height, width)) - inout
                    
                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                observation_coordinates = tuple(map(int, (norm_p[0]*width, norm_p[1]*height)))

                # NOTE that inout and raw_hm can reproduce norm_map if needed
                inference_data.append(
                    {
                        'frame': df.loc[i, 'frame'],
                        'observation_coordinates': observation_coordinates,
                        'raw_hm': raw_hm,
                        'inout': inout
                    }
                )
                
            with open(inference_filepath, "wb") as f:
                pickle.dump(inference_data, f)                