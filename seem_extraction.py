# --------------------------------------------------------
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), "dependencies/SEEM/demo_code"))

from tasks import *
import numpy as np
from rembg import remove
from config.cmdline import infer_image
import logging
import cv2
from dependencies.SEEM.demo_code.utils.constants import COCO_PANOPTIC_CLASSES
from dependencies.SEEM.demo_code.utils.arguments import load_opt_from_config_files
from dependencies.SEEM.demo_code.utils.distributed import init_distributed
from dependencies.SEEM.demo_code.xdecoder import build_model
from dependencies.SEEM.demo_code.xdecoder.BaseModel import BaseModel
import torch
from PIL import Image



@torch.no_grad()
def inference(model, image, reftxt):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return infer_image(model, image, reftxt)


def BuildSEEM() -> BaseModel:
    '''
    build args
    '''
    # args = parse_option()
    opt = load_opt_from_config_files(
        'dependencies/SEEM/demo_code/configs/seem/seem_focall_lang.yaml')
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join("model/seem_focall_v1.pt")
    if not os.path.exists(pretrained_pth):
        os.system("wget --directory-prefix=model/ {}".format(
            "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
    cur_model = 'Focal-L'

    '''
    build model
    '''
    model = BaseModel(opt, build_model(opt)).from_pretrained(
        pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

    return model


def SEEMPipeline(input_dir: str, output_dir: str, text_prompt: str) -> None:
    '''
    Main function for preprocessing section. This code will segment the background with
    rembg, and then use SEEM to segment the object based on the input text_prompt.

    Args: 
        input_dir: Input the absolute path for input image directory
        output_dir: Input the absolute path for directory to put the segmented image
    '''

    logging.info("Starting the SEEM pipeline")
    # check the output dir

    logging.info("Find the output directory")
    os.makedirs(output_dir, exist_ok=True)
    out_path = output_dir

    logging.info("Build SEEM")
    model = BuildSEEM()

    logging.info("start the task")
    with os.scandir(input_dir) as f:
        for entry in f:
            logging.info("open{}".format(entry.path))
            if entry.is_file():
                base_name, _ = os.path.splitext(entry.name)
                input_img = Image.open(entry.path)

                input_img = remove(input_img, bgcolor=(
                    0, 0, 0, 0)).convert('RGB')

                np_input = cv2.imread(entry.path)
                np_input = cv2.cvtColor(np_input, cv2.COLOR_BGR2BGRA)

                mask, pred_class = inference(
                    model=model, image=input_img, reftxt=text_prompt)
                logging.info("found this class{}".format(pred_class))

                mask = cv2.resize(
                    mask, (np_input.shape[1], np_input.shape[0]), interpolation=cv2.INTER_AREA)

                np_input[mask != 1] = 0

                logging.info("Output results")
                cv2.imwrite(os.path.join(out_path, base_name+'.png'), np_input)
            else:
                logging.warning("Input file included non-file")


def SEEMPreview(input_file: str, text_prompt: str) -> np.ndarray:
    '''
    This function allows the preview of the current picture.
    You can use this to see if the masking is working with the prompt you are using.

    Args:
        input_file: path to the image (ex: cwd/image.png)
        text_prompt: prompt to segment from the image
    Return:
        ndarray: uint8 type with values 0 or 255
    '''

    logging.info("Starting the SEEM Preview")
    # check the output dir

    logging.info("Build SEEM")
    model = BuildSEEM()

    logging.info("start the task")

    input_img = Image.open(input_file)

    input_img = remove(input_img, bgcolor=(0, 0, 0, 0)).convert('RGB')

    np_input = cv2.imread(input_file)
    np_input = cv2.cvtColor(np_input, cv2.COLOR_BGR2BGRA)

    mask, pred_class = inference(
        model=model, image=input_img, reftxt=text_prompt)
    logging.info("found this class{}".format(pred_class))

    mask = cv2.resize(
        mask, (np_input.shape[1], np_input.shape[0]), interpolation=cv2.INTER_AREA)
    mask = mask.astype(np.uint)*255

    logging.info("Output results")
    return mask
