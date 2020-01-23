import os
import os.path as osp
import scipy.io as sio
import numpy as np
from pycocotools.coco import COCO
from config import cfg
import json
import cv2
import random
import math
from utils.pose_utils import pixel2cam, process_bbox
from utils.vis import vis_keypoints, vis_3d_skeleton

class MuPoTS:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'MuPoTS', 'data', 'MultiPersonTestSet')
        self.test_annot_path = osp.join('..', 'data', 'MuPoTS', 'data', 'MuPoTS-3D.json')
        self.human_bbox_root_dir = osp.join('..', 'data', 'MuPoTS', 'bbox_root', 'bbox_root_mupots_output.json')
        self.joint_num = 21 # MuCo-3DHP
        self.joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe') # MuCo-3DHP
        self.original_joint_num = 17 # MuPoTS
        self.original_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head') # MuPoTS

        self.flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13) )
        self.skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7) )
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self.joints_have_depth = True
        self.root_idx = self.joints_name.index('Pelvis')
        self.data = self.load_data()

    def load_data(self):
        
        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0
        
        data = []
        db = COCO(self.test_annot_path)

        # use gt bbox and root
        if cfg.use_gt_info:
            print("Get bounding box and root from groundtruth")
            for aid in db.anns.keys():
                ann = db.anns[aid]
                if ann['is_valid'] == 0:
                    continue

                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);

                joint_cam = np.array(ann['keypoints_cam'])
                root_cam = joint_cam[self.root_idx]

                joint_img = np.array(ann['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)
                joint_img[:,2] = joint_img[:,2] - root_cam[2]
                joint_vis = np.ones((self.original_joint_num,1))
                
                bbox = np.array(ann['bbox'])
                img_width, img_height = img['width'], img['height']
                bbox = process_bbox(bbox, img_width, img_height)
                if bbox is None: continue
                
                data.append({
                    'img_path': img_path,
                    'bbox': bbox, 
                    'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'joint_vis': joint_vis,
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c,
                })
           
        else:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            
            for i in range(len(annot)):
                image_id = annot[i]['image_id']
                img = db.loadImgs(image_id)[0]
                img_width, img_height = img['width'], img['height']
                img_path = osp.join(self.img_dir, img['file_name'])
                fx, fy, cx, cy = img['intrinsic']
                f = np.array([fx, fy]); c = np.array([cx, cy]);
                root_cam = np.array(annot[i]['root_cam']).reshape(3)
                bbox = np.array(annot[i]['bbox']).reshape(4)

                data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'joint_img': np.zeros((self.original_joint_num, 3)), # dummy
                    'joint_cam': np.zeros((self.original_joint_num, 3)), # dummy
                    'joint_vis': np.zeros((self.original_joint_num, 1)), # dummy
                    'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                    'f': f,
                    'c': c,
                })

        return data

    def evaluate(self, preds, result_dir):
        
        print('Evaluation start...')
        gts = self.data
        sample_num = len(preds)
        joint_num = self.original_joint_num
 
        pred_2d_save = {}
        pred_3d_save = {}
        for n in range(sample_num):
            
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            img_name = gt['img_path'].split('/')
            img_name = img_name[-2] + '_' + img_name[-1].split('.')[0] # e.g., TS1_img_0001
            
            # restore coordinates to original space
            pred_2d_kpt = preds[n].copy()
            # only consider eval_joint
            pred_2d_kpt = np.take(pred_2d_kpt, self.eval_joint, axis=0)
            pred_2d_kpt[:,0] = pred_2d_kpt[:,0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_2d_kpt[:,1] = pred_2d_kpt[:,1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:,2] = (pred_2d_kpt[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + gt_3d_root[2]

            # 2d kpt save
            if img_name in pred_2d_save:
                pred_2d_save[img_name].append(pred_2d_kpt[:,:2])
            else:
                pred_2d_save[img_name] = [pred_2d_kpt[:,:2]]

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1,500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3,joint_num))
                tmpkps[0,:], tmpkps[1,:] = pred_2d_kpt[:,0], pred_2d_kpt[:,1]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)
            
            # 3d kpt save
            if img_name in pred_3d_save:
                pred_3d_save[img_name].append(pred_3d_kpt)
            else:
                pred_3d_save[img_name] = [pred_3d_kpt]
        
        output_path = osp.join(result_dir,'preds_2d_kpt_mupots.mat')
        sio.savemat(output_path, pred_2d_save)
        print("Testing result is saved at " + output_path)
        output_path = osp.join(result_dir,'preds_3d_kpt_mupots.mat')
        sio.savemat(output_path, pred_3d_save)
        print("Testing result is saved at " + output_path)

