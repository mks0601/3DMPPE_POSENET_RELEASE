import os
import os.path as osp
import json
import numpy as np
from pycocotools.coco import COCO

def calculate_score(output_path, annot_dir, subject_list):
    joint_num = 17 
    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    
    # aggregate annotations from each subject
    db = COCO()
    for subject in subject_list:
        with open(osp.join(annot_dir, 'Human36M_subject' + str(subject) + '.json'),'r') as f:
            annot = json.load(f)
        if len(db.dataset) == 0:
            for k,v in annot.items():
                db.dataset[k] = v
        else:
            for k,v in annot.items():
                db.dataset[k] += v
    db.createIndex()
        
    with open(output_path,'r') as f:
        output = json.load(f)
    
    p1_error = np.zeros((len(output), joint_num, 3)) # protocol #1 error (PA MPJPE)
    p2_error = np.zeros((len(output), joint_num, 3)) # protocol #2 error (MPJPE)
    p1_error_action = [ [] for _ in range(len(action_name)) ] # PA MPJPE for each action
    p2_error_action = [ [] for _ in range(len(action_name)) ] # MPJPE error for each action
    for n in range(len(output)):
        img_id = output[n]['image_id']

        pose_3d_out = np.array(output[n]['joint_cam']) # root_relative
        pose_3d_aligned_out = np.array(output[n]['joint_cam_aligned']) # aligned on root-relative gt
        
        gt_ann_id = db.getAnnIds(imgIds=[img_id])
        gt_ann = db.loadAnns(gt_ann_id)[0]
        pose_3d_gt = np.array(gt_ann['keypoints_cam'])
        root_idx = 0
        pose_3d_gt = pose_3d_gt - pose_3d_gt[root_idx] # root-relative gt
        
        p1_error[n] = np.power(pose_3d_aligned_out - pose_3d_gt,2) # PA MPJPE (protocol #1)
        p2_error[n] = np.power(pose_3d_out - pose_3d_gt,2)  # MPJPE (protocol #2)

        img = db.loadImgs([img_id])[0]
        img_name = img['file_name']
        action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
        p1_error_action[action_idx].append(p1_error[n].copy())
        p2_error_action[action_idx].append(p2_error[n].copy())

    # total error
    p1_tot_err = np.mean(np.power(np.sum(p1_error,axis=2),0.5))
    p2_tot_err = np.mean(np.power(np.sum(p2_error,axis=2),0.5))
    p1_eval_summary = 'Protocol #1 error (PA MPJPE) >> tot: %.2f\n' % (p1_tot_err)
    p2_eval_summary = 'Protocol #2 error (MPJPE) >> tot: %.2f\n' % (p2_tot_err)
   
    # error for each action
    for i in range(len(p1_error_action)):
        err = np.array(p1_error_action[i])
        err = np.mean(np.power(np.sum(err,axis=2),0.5))
        p1_eval_summary += (action_name[i] + ': %.2f ' % err)
    for i in range(len(p2_error_action)):
        err = np.array(p2_error_action[i])
        err = np.mean(np.power(np.sum(err,axis=2),0.5))
        p2_eval_summary += (action_name[i] + ': %.2f ' % err)
       
    print(p1_eval_summary)
    print()
    print(p2_eval_summary)


if __name__ == '__main__':
    output_path = './bbox_root_pose_human36m_output.json'
    annot_dir = './data/annotations'
    calculate_score(output_path, annot_dir, [9,11])

