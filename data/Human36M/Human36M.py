import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from config import cfg
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, rigid_align, process_bbox
import cv2
import random
import json
from utils.vis import vis_keypoints, vis_3d_skeleton

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
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16) # exclude Thorax

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
        print('Load data of H36M Protocol ' + str(self.protocol))

        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
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
            img_path = osp.join(self.img_dir, img['file_name'])
            img_width, img_height = img['width'], img['height']
           
            # check subject and frame_idx
            subject = img['subject']; frame_idx = img['frame_idx'];
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
                
            # project world coordinate to cam, image coordinate space
            action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_world = self.add_thorax(joint_world)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
            joint_img[:,2] = joint_img[:,2] - joint_cam[self.root_idx,2]
            joint_vis = np.ones((self.joint_num,1))
            
            if self.data_split == 'test' and not cfg.use_gt_info:
                bbox = bbox_root_result[str(image_id)]['bbox'] # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_cam = bbox_root_result[str(image_id)]['root']
            else:
                bbox = process_bbox(np.array(ann['bbox']), img_width, img_height)
                if bbox is None: continue
                root_cam = joint_cam[self.root_idx]
               
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
        
        pred_save = []
        error = np.zeros((sample_num, self.joint_num-1)) # joint error
        error_action = [ [] for _ in range(len(self.action_name)) ] # error for each sequence
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
            pred_2d_kpt[:,0] = pred_2d_kpt[:,0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_2d_kpt[:,1] = pred_2d_kpt[:,1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            pred_2d_kpt[:,2] = (pred_2d_kpt[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + gt_3d_root[2]

            vis = False
            if vis:
                cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                filename = str(random.randrange(1,500))
                tmpimg = cvimg.copy().astype(np.uint8)
                tmpkps = np.zeros((3,self.joint_num))
                tmpkps[0,:], tmpkps[1,:] = pred_2d_kpt[:,0], pred_2d_kpt[:,1]
                tmpkps[2,:] = 1
                tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                cv2.imwrite(filename + '_output.jpg', tmpimg)

            # back project to camera coordinate system
            pred_3d_kpt = pixel2cam(pred_2d_kpt, f, c)
 
            # root joint alignment
            pred_3d_kpt = pred_3d_kpt - pred_3d_kpt[self.root_idx]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[self.root_idx]
           
            if self.protocol == 1:
                # rigid alignment for PA MPJPE (protocol #1)
                pred_3d_kpt = rigid_align(pred_3d_kpt, gt_3d_kpt)
            
            # exclude thorax
            pred_3d_kpt = np.take(pred_3d_kpt, self.eval_joint, axis=0)
            gt_3d_kpt = np.take(gt_3d_kpt, self.eval_joint, axis=0)
           
            # error calculate
            error[n] = np.sqrt(np.sum((pred_3d_kpt - gt_3d_kpt)**2,1))
            img_name = gt['img_path']
            action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
            error_action[action_idx].append(error[n].copy())

            # prediction save
            pred_save.append({'image_id': image_id, 'joint_cam': pred_3d_kpt.tolist(), 'bbox': bbox.tolist(), 'root_cam': gt_3d_root.tolist()}) # joint_cam is root-relative coordinate

        # total error
        tot_err = np.mean(error)
        metric = 'PA MPJPE' if self.protocol == 1 else 'MPJPE'
        eval_summary = 'Protocol ' + str(self.protocol) + ' error (' + metric + ') >> tot: %.2f\n' % (tot_err)

        # error for each action
        for i in range(len(error_action)):
            err = np.mean(np.array(error_action[i]))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
           
        print(eval_summary)

        # prediction save
        output_path = osp.join(result_dir, 'bbox_root_pose_human36m_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)

