from __future__ import print_function
import sys
import os
import glob
import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from tqdm import tqdm

import sys
sys.path.append('/proj/tools/Pytorch_Retinaface')
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer

import orientation_calculator
from dataset_loader import DatasetLoader


parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model',
                    default='/proj/tools/Pytorch_Retinaface/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='/proj/brizk/output/retinaface', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.60, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


    
if __name__ == '__main__':

    if args.save_image:
        imgs_save_folder = args.save_folder + "/imgs"
        if not os.path.isdir(imgs_save_folder):
            os.makedirs(imgs_save_folder)
            print("creating imgs directory")
    
    parent_directory =\
        "/".join(args.save_folder.split('/')[:-1] + ['imgs'])
    if not os.path.isdir(parent_directory):
        os.makedirs(parent_directory)


    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    dataset = DatasetLoader()

    skipped_videos = []

    logger_file = open(f"{parent_directory}/recent_run_log.txt", "w")


    # testing begin
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    print("DEBUG: current working directory is", os.getcwd())
    for video_num, video in enumerate(tqdm(dataset)):
        logger_file.write(f'starting video {video_num}: {dataset.current_ds}/{video.name}')
        logger_file.flush()

        video_save_folder =\
            f'{args.save_folder}/{dataset.current_ds}/{video.name}'
        if os.path.isdir(video_save_folder):
            print(f'Skipping {video_save_folder} as already exists')
            logger_file.writeline(f'Skipping {video_save_folder} as already exists')
            logger_file.flush()
            continue

        os.makedirs(video_save_folder)

        video_frames_folder =\
            f'{parent_directory}/{dataset.current_ds}/{video.name}'

        if not os.path.isdir(video_frames_folder):
            os.makedirs(video_frames_folder)

        for i, img_raw in enumerate(tqdm(video)):
            img = np.float32(img_raw)

            # testing scale
            target_size = 1600
            max_size = 2150
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(im_size_min)
            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if args.origin_size:
                resize = 1

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            _t['forward_pass'].tic()
            loc, conf, landms = net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()

            # --------------------------------------------------------------------
            if len(dets) <= 0:
                continue
            
            cv2.imwrite(f'{video_frames_folder}/{i}.jpg', img_raw)

            with open(f'{video_save_folder}/{i}.txt', "w") as fd:
                bboxs = dets
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(f"{dataset.current_ds} {video.name} {i}\n")
                fd.write(bboxs_num)
                
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)
                    line = ""
                    for v in box:
                        line += str(v) + " "
                    fd.write(line + "\n")


            # print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, video.num_of_imgs, _t['forward_pass'].average_time, _t['misc'].average_time))

            # save image
            if args.save_image:
                orientation_calculator.draw(
                    img_raw, dets,
                    f'{org_imgs_directory}/{i}',
                    args.vis_thres
                )
    
    logger_file.close()