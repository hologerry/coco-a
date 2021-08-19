import json
import time
import sys
import random

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import numpy as np
import copy
import cv2


# drawing imports
import skimage.io as io


# directory containing coco-a annotations
COCOA_DIR = '/D_data/Seg/data/cocoa/annotations'
# coco-a json file
COCOA_ANN = 'cocoa_beta2015.json'
# directory containing VisualVerbnet
VVN_DIR = '/D_data/Seg/data/cocoa/annotations'
# vvn json file
VVN_ANN = 'visual_verbnet_beta2015.json'
# directory containing the MS COCO images
COCO_IMG_DIR = '/D_data/Seg/data/coco/images'
# directory containing the MS COCO Python API
COCO_API_DIR = '/D_data/Seg/cocoapi/PythonAPI'
# directory containing the MS COCO annotations
COCO_ANN_DIR = '/D_data/Seg/data/coco/annotations'

# load coco annotations
ANN_FILE_PATH = "{0}/instances_{1}.json".format(COCO_ANN_DIR,'train2014cocoa')

# coco = COCO( ANN_FILE_PATH )


with open(ANN_FILE_PATH, 'r') as f:
    dataset = json.load(f)

new_dataset = copy.deepcopy(dataset)

new_dataset['annotations'] = []

for ann in dataset['annotations']:
    if ann['category_id'] == 1:
        new_dataset['annotations'].append(ann)

NEW_ANN_FILE_PATH = "{0}/instances_{1}.json".format(COCO_ANN_DIR,'train2014')

with open(NEW_ANN_FILE_PATH, 'w') as f:
    json.dump(new_dataset, f)
    print(f"saved new annotations file to {NEW_ANN_FILE_PATH}")
