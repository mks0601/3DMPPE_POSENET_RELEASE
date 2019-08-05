import os
import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
import json
import cv2
import random
import math

annot_path = osp.join('coco', 'person_keypoints_val2017.json')

data = []
db = COCO(annot_path)
fp = open('coco_img_name.txt','w') 
for iid in db.imgs.keys():
    img = db.imgs[iid]
    imgname = img['file_name']
    imgname = 'coco_' + imgname.split('.')[0]
    fp.write(imgname + '\n')
fp.close()

