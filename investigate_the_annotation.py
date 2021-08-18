import json
from pycocotools.coco import COCO

# ann_path = '/D_data/Seg/data/coco/annotations/instances_train2014coco.json'
ann_path = '/D_data/Seg/data/coco/annotations/instances_train2014.json'


with open(ann_path, 'r') as f:
    dataset = json.load(f)

print(dataset['annotations'][:10])
# coco = COCO(ann_path)
