import os
import os.path as osp
import scipy.io as sio
import numpy as np
import cv2
import random
import json
import math
from tqdm import tqdm

root_dir = './images' # define path here
save_dir = './annotations' # define path here

joint_num = 17
subject_list = [1, 5, 6, 7, 8, 9, 11]
action_idx = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
subaction_idx = (1, 2)
camera_idx = (1, 2, 3, 4)
action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

def load_h36m_annot_file(annot_file):
    data = sio.loadmat(annot_file)
    joint_world = data['pose3d_world'] # 3D world coordinates of keypoints
    R = data['R'] # extrinsic
    T = np.reshape(data['T'],(3)) # extrinsic
    f = np.reshape(data['f'],(-1)) # focal legnth
    c = np.reshape(data['c'],(-1)) # principal points
    img_heights = np.reshape(data['img_height'],(-1))
    img_widths = np.reshape(data['img_width'],(-1))
   
    return joint_world, R, T, f, c, img_widths, img_heights

def _H36FolderName(subject_id, act_id, subact_id, camera_id):
    return "s_%02d_act_%02d_subact_%02d_ca_%02d" % \
           (subject_id, act_id, subact_id, camera_id)

def _H36ImageName(folder_name, frame_id):
    return "%s_%06d.jpg" % (folder_name, frame_id + 1)

def cam2pixel(cam_coord, f, c):
    x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
    y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
    return x,y

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord - t)
    return cam_coord

def get_bbox(joint_img):
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:,0])
    ymin = np.min(joint_img[:,1])
    xmax = np.max(joint_img[:,0])
    ymax = np.max(joint_img[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    bbox[0] = (xmin + xmax)/2. - width/2*1.2
    bbox[1] = (ymin + ymax)/2. - height/2*1.2
    bbox[2] = width*1.2
    bbox[3] = height*1.2

    return bbox

img_id = 0; annot_id = 0
for subject in tqdm(subject_list):
    cam_param = {}
    joint_3d = {}
    images = []; annotations = [];
    for aid in tqdm(action_idx):
        for said in tqdm(subaction_idx):
            for cid in tqdm(camera_idx):
                folder = _H36FolderName(subject,aid,said,cid)
                if folder == 's_11_act_02_subact_02_ca_01':
                    continue
               
                joint_world, R, t, f, c, img_widths, img_heights = load_h36m_annot_file(osp.join(root_dir, folder, 'h36m_meta.mat'))

                if str(aid) not in joint_3d:
                    joint_3d[str(aid)] = {}
                if str(said) not in joint_3d[str(aid)]:
                    joint_3d[str(aid)][str(said)] = {}

                img_num = np.shape(joint_world)[0]
                for n in range(img_num):
                    img_dict = {}
                    img_dict['id'] = img_id
                    img_dict['file_name'] = osp.join(folder, _H36ImageName(folder, n))
                    img_dict['width'] = int(img_widths[n])
                    img_dict['height'] = int(img_heights[n])
                    img_dict['subject'] = subject
                    img_dict['action_name'] = action_name[aid-2]
                    img_dict['action_idx'] = aid
                    img_dict['subaction_idx'] = said
                    img_dict['cam_idx'] = cid
                    img_dict['frame_idx'] = n
                    images.append(img_dict)
                    
                    if str(cid) not in cam_param:
                        cam_param[str(cid)] = {'R': R.tolist(), 't': t.tolist(), 'f': f.tolist(), 'c': c.tolist()}
                    if str(n) not in joint_3d[str(aid)][str(said)]:
                        joint_3d[str(aid)][str(said)][str(n)] = joint_world[n].tolist()

                    annot_dict = {}
                    annot_dict['id'] = annot_id
                    annot_dict['image_id'] = img_id

                    # project world coordinate to cam, image coordinate space
                    joint_cam = np.zeros((joint_num,3))
                    for j in range(joint_num):
                        joint_cam[j] = world2cam(joint_world[n][j], R, t)
                    joint_img = np.zeros((joint_num,2))
                    joint_img[:,0], joint_img[:,1] = cam2pixel(joint_cam, f, c)
                    joint_vis = (joint_img[:,0] >= 0) * (joint_img[:,0] < img_widths[n]) * (joint_img[:,1] >= 0) * (joint_img[:,1] < img_heights[n])
                    annot_dict['keypoints_vis'] = joint_vis.tolist()
                    
                    bbox = get_bbox(joint_img)
                    annot_dict['bbox'] = bbox.tolist() # xmin, ymin, width, height
                    annotations.append(annot_dict)

                    img_id += 1
                    annot_id += 1
    
    data = {'images': images, 'annotations': annotations}
    with open(osp.join(save_dir, 'Human36M_subject' + str(subject) + '_data.json'), 'w') as f:
        json.dump(data, f)    
    with open(osp.join(save_dir, 'Human36M_subject' + str(subject) + '_camera.json'), 'w') as f:
        json.dump(cam_param, f)
    with open(osp.join(save_dir, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'w') as f:
        json.dump(joint_3d, f)
