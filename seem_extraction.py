# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import torch
import argparse
import whisper
import numpy as np

from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
import cv2

from rembg import remove, new_session


from tasks import *
print(os.path.join(os.getcwd(),'output/'))

def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/seem_focall_lang.yaml", metavar="FILE", help='path to config file', )
    args = parser.parse_args()

    return args

'''
build args
'''
args = parse_option()
opt = load_opt_from_config_files(args.conf_files)
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in args.conf_files:
    pretrained_pth = os.path.join("seem_focalt_v2.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focalt_v2.pt"))
    cur_model = 'Focal-T'
elif 'focal' in args.conf_files:
    pretrained_pth = os.path.join("seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget {}".format("https://projects4jw.blob.core.windows.net/x-decoder/release/seem_focall_v1.pt"))
    cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

'''
audio
'''
audio = whisper.load_model("base")

@torch.no_grad()
def inference(image, task, reftxt):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # if 'Video' in task:
        # return interactive_infer_video(model, audio, image, [''], reftxt='person')
        
        return interactive_infer_image(model, audio, image, task, reftxt=reftxt)
img_path = "../shoe/images/"
os.makedirs(os.getcwd()+'/output/', exist_ok=True)
out_path = os.path.join(os.getcwd(), 'output/')
print(os.getcwd())
cnt = 0
with os.scandir(img_path) as f:
    for entry in f:
        if entry.is_file():
            base_name, _ = os.path.splitext(entry.name)
            input_img = Image.open(entry.path)
            print(input_img.size)
            input_img = remove(input_img, bgcolor=(0,0,0,0)).convert('RGB')
            # print(type(input_img))
            np_input = cv2.imread(entry.path)

            print(np_input.shape)
            np_input = cv2.cvtColor(np_input, cv2.COLOR_BGR2BGRA)
            img, mask = inference({"image":input_img}, ['Text'], 'shoe')

            img = img.convert('RGB')
            for i, m in enumerate(mask):
                m = cv2.resize(m, (np_input.shape[1], np_input.shape[0]), interpolation = cv2.INTER_AREA)

                np_input[m==255] = 0
                cv2.imwrite(out_path+base_name+'.png', np_input)
                # cv2.imwrite(out_path+str(cnt)+'_mask'+'_'+str(i)+'.png', m)
            # mask = mask.convert('RGB')
            # img.save(out_path+str(cnt),'png')
            # mask.save(out_path+str(cnt)+'_mask', 'jpeg')
            cnt+=1
