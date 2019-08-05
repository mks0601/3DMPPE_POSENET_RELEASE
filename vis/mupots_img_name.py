import os
import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
import json
import cv2
import random
import math

annot_path = osp.join('mupots', 'MuPoTS-3D.json')

data = []
db = COCO(annot_path)
fp = open('mupots_img_name.txt','w') 
for iid in db.imgs.keys():
    img = db.imgs[iid]
    imgname = img['file_name'].split('/')
    folder_id = int(imgname[0][2:])
    frame_id = int(imgname[1].split('.')[0][4:])
    fp.write(str(folder_id) + ' ' + str(frame_id) + '\n')
fp.close()

