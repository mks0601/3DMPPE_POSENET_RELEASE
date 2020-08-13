import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_pose_net
from dataset import generate_patch_image
from utils.pose_utils import process_bbox, pixel2cam
from utils.vis import vis_keypoints, vis_3d_multiple_skeleton

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# MuCo joint set
joint_num = 21
joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
flip_pairs = ( (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20) )
skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_pose_net(cfg, False, joint_num)
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'])
model.eval()

# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
img_path = 'input.jpg'
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox_list = [
[139.41, 102.25, 222.39, 241.57],\
[287.17, 61.52, 74.88, 165.61],\
[540.04, 48.81, 99.96, 223.36],\
[372.58, 170.84, 266.63, 217.19],\
[0.5, 43.74, 90.1, 220.09]] # xmin, ymin, width, height
root_depth_list = [11250.5732421875, 15522.8701171875, 11831.3828125, 8852.556640625, 12572.5966796875] # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
assert len(bbox_list) == len(root_depth_list)
person_num = len(bbox_list)

# normalized camera intrinsics
focal = [1500, 1500] # x-axis, y-axis
princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

# for each cropped and resized human image, forward it to PoseNet
output_pose_2d_list = []
output_pose_3d_list = []
for n in range(person_num):
    bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
    img, img2bb_trans = generate_patch_image(original_img, bbox, False, 1.0, 0.0, False) 
    img = transform(img).cuda()[None,:,:,:]

    # forward
    with torch.no_grad():
        pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

    # inverse affine transform (restore the crop and resize)
    pose_3d = pose_3d[0].cpu().numpy()
    pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
    pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
    pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
    img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
    pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
    output_pose_2d_list.append(pose_3d[:,:2].copy())
    
    # root-relative discretized depth -> absolute continuous depth
    pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list[n]
    pose_3d = pixel2cam(pose_3d, focal, princpt)
    output_pose_3d_list.append(pose_3d.copy())

# visualize 2d poses
vis_img = original_img.copy()
for n in range(person_num):
    vis_kps = np.zeros((3,joint_num))
    vis_kps[0,:] = output_pose_2d_list[n][:,0]
    vis_kps[1,:] = output_pose_2d_list[n][:,1]
    vis_kps[2,:] = 1
    vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
cv2.imwrite('output_pose_2d.jpg', vis_img)

# visualize 3d poses
vis_kps = np.array(output_pose_3d_list)
vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')
