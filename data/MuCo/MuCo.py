import os
import os.path as osp
import numpy as np
import math
from utils.pose_utils import process_bbox
from pycocotools.coco import COCO
from config import cfg

class MuCo:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MuCo', 'data')
        self.train_annot_path = osp.join('..', 'data', 'MuCo', 'data', 'MuCo-3DHP.json')
        self.joint_num = 21
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
        self.joints_have_depth = True
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):

        if self.data_split == 'train':
            db = COCO(self.train_annot_path)
        else:
            print('Unknown data subset')
            assert 0

        data = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            img_path = osp.join(self.img_dir, imgname)
            f = img["f"]
            c = img["c"]

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)

            root_depths = [ann['keypoints_cam'][self.root_idx][2] for ann in anns]
            closest_pid = root_depths.index(min(root_depths))
            pid_list = [closest_pid]
            for i in range(len(anns)):
                if i == closest_pid:
                    continue
                picked = True
                for j in range(len(anns)):
                    if i == j:
                        continue
                    dist = (np.array(anns[i]['keypoints_cam'][self.root_idx]) - np.array(anns[j]['keypoints_cam'][self.root_idx])) ** 2
                    dist_2d = math.sqrt(np.sum(dist[:2]))
                    dist_3d = math.sqrt(np.sum(dist))
                    if dist_2d < 500 or dist_3d < 500:
                        picked = False
                if picked:
                    pid_list.append(i)
            
            for pid in pid_list:
                joint_cam = np.array(anns[pid]['keypoints_cam'])
                root_cam = joint_cam[self.root_idx]
                
                joint_img = np.array(anns[pid]['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_img[:,2] = joint_img[:,2] - root_cam[2]
                joint_vis = np.ones((self.joint_num,1))

                bbox = process_bbox(anns[pid]['bbox'], img_width, img_height)
                if bbox is None: continue

                data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'joint_vis': joint_vis,
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c
                })


        return data


