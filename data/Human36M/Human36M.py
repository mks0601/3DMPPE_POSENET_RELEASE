import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from config import cfg
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, rigid_align, get_bbox, warp_coord_to_original
import cv2
import random
import json
from utils.vis import vis_keypoints, vis_3d_skeleton
from Human36M_eval import calculate_score

class Human36M:
    def __init__(self, data_split):
        self.data_split = data_split
        self.img_dir = osp.join('..', 'data', 'Human36M', 'images')
        self.annot_path = osp.join('..', 'data', 'Human36M', 'annotations')
        self.human_bbox_root_dir = osp.join('..', 'data', 'Human36M', 'bbox_root', 'bbox_root_human36m_output.json')
        self.joint_num = 18 # original:17, but manually added 'Thorax'
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        self.flip_pairs = ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) )
        self.skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.joints_have_depth = True
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16) # except Thorax

        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.root_idx = self.joints_name.index('Pelvis')
        self.lshoulder_idx = self.joints_name.index('L_Shoulder')
        self.rshoulder_idx = self.joints_name.index('R_Shoulder')
        self.protocol = 2
        self.data = self.load_data()

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1,5,6,7,8,9]
            elif self.protocol == 2:
                subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def add_thorax(self, joint_coord):
        thorax = (joint_coord[self.lshoulder_idx, :] + joint_coord[self.rshoulder_idx, :]) * 0.5
        thorax = thorax.reshape((1, 3))
        joint_coord = np.concatenate((joint_coord, thorax), axis=0)
        return joint_coord

    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        for subject in subject_list:
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
        db.createIndex()
        
        if self.data_split == 'test' and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}
        else:
            print("Get bounding box and root from groundtruth")

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            
            # check subject and frame_idx
            subject = img['subject']; frame_idx = img['frame_idx'];
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
            cam_param = img['cam_param']
            R,t,f,c = np.array(cam_param['R']), np.array(cam_param['t']), np.array(cam_param['f']), np.array(cam_param['c'])
                
            # project world coordinate to cam, image coordinate space
            joint_cam = np.array(ann['keypoints_cam'])
            joint_cam = self.add_thorax(joint_cam)
            joint_img = np.zeros((self.joint_num,3))
            joint_img[:,0], joint_img[:,1], joint_img[:,2] = cam2pixel(joint_cam, f, c)
            joint_img[:,2] = joint_img[:,2] - joint_cam[self.root_idx,2]
            joint_vis = np.ones((self.joint_num,1))
            
            if self.data_split == 'test' and not cfg.use_gt_info:
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_cam = bbox_root_result[str(image_id)]['root']
            else:
                bbox = np.array(ann['bbox'])
                root_cam = joint_cam[self.root_idx]
               
                # aspect ratio preserving bbox
                w = bbox[2]
                h = bbox[3]
                c_x = bbox[0] + w/2.
                c_y = bbox[1] + h/2.
                aspect_ratio = cfg.input_shape[1]/cfg.input_shape[0]
                if w > aspect_ratio * h:
                    h = w / aspect_ratio
                elif w < aspect_ratio * h:
                    w = h * aspect_ratio
                bbox[2] = w*1.25
                bbox[3] = h*1.25
                bbox[0] = c_x - bbox[2]/2.
                bbox[1] = c_y - bbox[3]/2.

            data.append({
                'img_path': img_path,
                'img_id': image_id,
                'bbox': bbox,
                'joint_img': joint_img, # [org_img_x, org_img_y, depth - root_depth]
                'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                'joint_vis': joint_vis,
                'root_cam': root_cam, # [X, Y, Z] in camera coordinate
                'f': f,
                'c': c})
           
        return data

    def evaluate(self, preds, result_dir):
        
        print('Evaluation start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)
        joint_num = self.joint_num
        
        pred_save = []
        for n in range(sample_num):
            gt = gts[n]
            image_id = gt['img_id']
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_3d_root = gt['root_cam']
            gt_3d_kpt = gt['joint_cam']
            gt_vis = gt['joint_vis']
            
            # restore coordinates to original space
            pred_2d_kpt = preds[n].copy()
            pred_2d_kpt[:,0], pred_2d_kpt[:,1], pred_2d_kpt[:,2] = warp_coord_to_original(pred_2d_kpt, bbox, gt_3d_root)
            
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
            pred_3d_kpt = np.zeros((joint_num,3))
            pred_3d_kpt[:,0], pred_3d_kpt[:,1], pred_3d_kpt[:,2] = pixel2cam(pred_2d_kpt, f, c)

            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[self.root_idx]

            # rigid alignment for PA MPJPE (protocol #1)
            pred_3d_kpt_align = rigid_align(pred_3d_kpt, gt_3d_kpt)
            
            # exclude thorax
            pred_3d_kpt = np.take(pred_3d_kpt, self.eval_joint, axis=0)
            pred_3d_kpt_align = np.take(pred_3d_kpt_align, self.eval_joint, axis=0)

            # prediction save
            pred_save.append({'image_id': image_id, 'joint_cam': pred_3d_kpt.tolist(), 'joint_cam_aligned': pred_3d_kpt_align.tolist(), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()}) # joint_cams are root-relative

        output_path = osp.join(result_dir, 'bbox_root_pose_human36m_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

        calculate_score(output_path, self.annot_path, self.get_subject())
