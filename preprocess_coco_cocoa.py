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



# load cocoa annotations

print("Loading COCO-a annotations...")
tic = time.time()

with open("{0}/{1}".format(COCOA_DIR,COCOA_ANN)) as f:
    cocoa = json.load(f)

# annotations with agreement of at least 1 mturk annotator
cocoa_1 = cocoa['annotations']['1']
# annotations with agreement of at least 2 mturk annotator
cocoa_2 = cocoa['annotations']['2']
# annotations with agreement of at least 3 mturk annotator
cocoa_3 = cocoa['annotations']['3']

print("Done, (t={0:.2f}s).".format(time.time() - tic))


# load visual verbnet

print("Loading VisualVerbNet...")
tic = time.time()

with open("{0}/{1}".format(VVN_DIR,VVN_ANN)) as f:
    vvn = json.load(f)

# list of 145 visual actions contained in VVN
visual_actions = vvn['visual_actions']
# list of 17 visual adverbs contained in VVN
visual_adverbs = vvn['visual_adverbs']

print("Done, (t={0:.2f}s).".format(time.time() - tic))


# visual actions in VVN by category

# each visual action is a dictionary with the following properties:
#  - id:            unique id within VVN
#  - name:          name of the visual action
#  - category:      visual category as defined in the paper
#  - definition:    [empty]
#                   an english language description of the visual action
#  - verbnet_class: [empty]
#                   corresponding verbnet (http://verbs.colorado.edu/verb-index/index.php) entry id for each visual action

for cat in set([x['category'] for x in visual_actions]):
    print("Visual Category: [{0}]".format(cat))
    for va in [x for x in visual_actions if x['category']==cat]:
        print("\t - id:[{0}], visual_action:[{1}]".format(va['id'],va['name']))



# visual adverbs in VVN by category

# each visual adverb is a dictionary with the following properties:
#  - id:            unique id within VVN
#  - name:          name of the visual action
#  - category:      visual category as defined in the paper
#  - definition:    [empty]
#                   an english language description of the visual action

# NOTE: relative_location is the location of the object with respect to the subject.
# It is not with respect to the reference frame of the image.
# i.e. if you where the subject, where is the object with respect to you?

for cat in set([x['category'] for x in visual_adverbs]):
    print("Visual Category: [{0}]".format(cat))
    for va in [x for x in visual_adverbs if x['category']==cat]:
        print("\t - id:[{0}], visual_adverb:[{1}]".format(va['id'],va['name']))




# coco-a is organized to be easily integrable with MS COCO

# load coco annotations
ANN_FILE_PATH = "{0}/instances_{1}.json".format(COCO_ANN_DIR,'train2014coco')

coco = COCO( ANN_FILE_PATH )


# https://zhuanlan.zhihu.com/p/84214563
def mask2polygon(mask):
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation


def interact_condition(x, image_id):
    if x['image_id'] != image_id:
        return False
    if x['object_id'] == -1:  # not interacting
        return False
    if 15 not in x['visual_adverbs']:  # not full contact
        return False
    if coco.loadAnns(x['object_id'])[0]['category_id'] == 1:  # ignore person
        return False
    return True

all_image_ids = coco.getImgIds()

new_image_id_to_anns = {}

for idx, image_id in enumerate(all_image_ids):
    raw_image_anns = coco.imgToAnns[image_id]
    new_image_anns = copy.deepcopy(raw_image_anns)

    interactions = [x for x in cocoa_2 if interact_condition(x, image_id)]
    img = coco.loadImgs(image_id)[0]

    subject_id_to_interactions = {}
    for interaction in interactions:
        subject_id = interaction['subject_id']
        if subject_id not in subject_id_to_interactions:
            subject_id_to_interactions[subject_id] = []
        subject_id_to_interactions[subject_id].append(interaction)

    for subject_id, subject_interactions in subject_id_to_interactions.items():

        subject_ann = coco.loadAnns(subject_id)[0]
        new_subject_ann = copy.deepcopy(subject_ann)
        if subject_ann not in new_image_anns:
            print("subject_id", subject_id)
            print("subject_ann", subject_ann)
            print("subject_interactions", subject_interactions)
            print("raw_image_anns", raw_image_anns)
            print("new_image_anns", new_image_anns)
        new_image_anns.remove(subject_ann)

        for sub_interact in subject_interactions:  # all interactions related to current subject

            object_id = sub_interact['object_id']
            object_ann = coco.loadAnns(object_id)[0]
            # object_cat = coco.cats[object_ann['category_id']]['name']

            # # maybe it is already full contacted by other person, just ignore
            # if object_ann not in new_image_anns:
            #     print("object_id", object_id)
            #     print("object_ann", object_ann)
            #     print("subject_interactions", subject_interactions)
            #     print("raw_image_anns", raw_image_anns)
            #     print("new_image_anns", new_image_anns)

            if object_ann in new_image_anns:
                new_image_anns.remove(object_ann)

                new_subject_ann['segmentation'] += object_ann['segmentation']

        rles = maskUtils.frPyObjects(new_subject_ann['segmentation'], img['height'], img['width'])
        rle = maskUtils.merge(rles)

        m = maskUtils.decode(rle)

        new_subject_ann['segmentation'] = mask2polygon(m)
        new_subject_ann['area'] = float(maskUtils.area(rle))
        print("new_subject_ann['area']", new_subject_ann['area'])
        assert type(new_subject_ann['area']) == float
        new_subject_ann['bbox'] = maskUtils.toBbox(rle).tolist()
        print("new_subject_ann['bbox']", new_subject_ann['bbox'])
        assert type(new_subject_ann['bbox']) == list and len(new_subject_ann['bbox']) == 4

        new_image_anns.append(new_subject_ann)

    new_image_id_to_anns[image_id] = new_image_anns

    if (idx + 1) % 100 == 0:
        print(f"processed [{idx+1}/{len(all_image_ids)}]")



with open(ANN_FILE_PATH, 'r') as f:
    dataset = json.load(f)

dataset['annotations'] = []
for k, v in new_image_id_to_anns.items():
    dataset['annotations'] += v

NEW_ANN_FILE_PATH = "{0}/instances_{1}.json".format(COCO_ANN_DIR,'train2014')

with open(NEW_ANN_FILE_PATH, 'w') as f:
    json.dump(dataset, f)
    print(f"saved new annotations file to {NEW_ANN_FILE_PATH}")
