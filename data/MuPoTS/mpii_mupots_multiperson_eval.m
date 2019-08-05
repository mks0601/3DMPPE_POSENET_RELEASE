function mpii_mupots_multiperson_eval(eval_mode, is_relative)

% eval_mode: EVLAUATION_MODE
% is_relative: 1: root-relative 3D multi-person pose estimation, 0: absolute 3D multi-person pose estimation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outline of the test eval procedure on MuPoTS-3D. 
% Plug in your predictions at the appropriate point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mpii_mupots_config;
addpath('./util');
[~,o1,o2,relevant_labels] = mpii_get_joints('relevant');  
num_joints = length(o1);

%Path to the test images and annotations
test_annot_base = mpii_mupots_path; %See mpii_mupots_config
%Path where results are written out 
results_output_path = './';

%If predicted joints have a different ordering, specify mapping to MPI joints here
%map_to_mpii_jointset = % [11 14 10 13 9 12 5 8 4 7 3 6 1];
%Order to process bones in to resize them to the GT
safe_traversal_order = [15, 16, 2, 1, 17, 3, 4, 5, 6, 7, 8, 9:14];

EVALUATION_MODE = eval_mode; % 0 = evaluate all annotated persons, 1 = evaluate only predictions matched to annotations

person_colors = {'red', 'yellow', 'green', 'blue', 'magenta', 'cyan', 'black', 'white'} ;

sequencewise_per_joint_error = {};
sequencewise_undetected_people = [];
sequencewise_visibility_mask = {};
sequencewise_occlusion_mask = {};
sequencewise_annotated_people = [];
sequencewise_frames = [];

%% load prdictions
preds_2d_kpt = load('preds_2d_kpt_mupots.mat');
preds_3d_kpt = load('preds_3d_kpt_mupots.mat');

for ts = 1:20
    person_ids = [];
    open_person_ids = 1:20;

    load( sprintf('%s/TS%d/annot.mat',test_annot_base, ts));
    load( sprintf('%s/TS%d/occlusion.mat',test_annot_base, ts));
    
    num_frames = size(annotations,1);
    
    undetected_people = 0;
    annotated_people = 0;
    pje_idx = 1;
    
    per_joint_error = []; %zeros(17,1,num_test_points);
    per_joint_occlusion_mask = [];
    per_joint_visibility_mask = [];
    sequencewise_frames(ts) = num_frames;

for i = 1:num_frames

     %Count valid annotations
     valid_annotations = 0;
     for k = 1:size(annotations,2)
         if(annotations{i,k}.isValidFrame)
             valid_annotations = valid_annotations + 1;
         end
     end
     annotated_people = annotated_people + valid_annotations;
     
     if(valid_annotations == 0)
         continue;
     end
     
     gt_pose_2d =  cell(valid_annotations,1);
     gt_pose_3d =  cell(valid_annotations,1);
     gt_visibility = cell(valid_annotations,1);
     gt_pose_occlusion_labels =  cell(valid_annotations,1);
     gt_pose_visibility_labels =  cell(valid_annotations,1);
     %The joint set to use for matching predictions to GT
     matching_joints = [2:14];
     %matching_joints = [2 3 6 9 12];

     idx = 1;
     for k = 1:size(annotations,2)
         if(annotations{i,k}.isValidFrame)
             
             gt_pose_2d{idx} = annotations{i,k}.annot2(:,matching_joints); 
             gt_pose_3d{idx} = annotations{i,k}.univ_annot3 ;
             gt_visibility{idx} = ones(1,length(matching_joints));
             gt_pose_occlusion_labels{idx} = occlusion_labels{i,k} ;
             gt_pose_visibility_labels{idx} = 1 - occlusion_labels{i,k} ;
             idx = idx + 1;
         end
     end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Predictions here     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %img = imread(sprintf('%s/TS%d/img_%06d.jpg',test_annot_base, ts, i-1));
    
    % prediction of this image
    pred_2d_kpt = getfield(preds_2d_kpt,sprintf('TS%d_img_%06d',ts, i-1));
    pred_3d_kpt = getfield(preds_3d_kpt,sprintf('TS%d_img_%06d',ts, i-1));

    %Number of subjects predicted 
    num_pred = size(pred_2d_kpt,1);

    pred_pose_2d = cell(num_pred,1);
    pred_pose_3d = cell(num_pred,1);
    pred_visibility = cell(num_pred,1);
     for k = 1:num_pred
         
         pred_pose_2d{k} = zeros(2,14);
         %pred_pose_2d{k}(:,map_to_mpii_jointset) = % 2D Pose for person detected person k;
         pred_pose_2d{k} = transpose(squeeze(pred_2d_kpt(k,:,:))); % 2D Pose for person detected person k;

         % If some joints such as neck are missing, they can be estimated as the mean of shoulders
         %pred_pose_2d{k}(:,2) = mean(pred_pose_2d{k}(:,[3,6]),2);

         pred_pose_2d{k} = pred_pose_2d{k}(:,matching_joints);
         pred_visibility{k} = ~((pred_pose_2d{k}(1,:) == 0) & (pred_pose_2d{k}(2,:) == 0));
         
         pred_pose_3d{k} = zeros(3,num_joints);
         %pred_pose_3d{k}(:,map_to_mpii_jointset) = % 3D Pose for person detected person k;
         pred_pose_3d{k} = transpose(squeeze(pred_3d_kpt(k,:,:))); % 3D Pose for person detected person k;

         % If some joints such as neck or pelvis are missing, they can be estimated as 
         % the mean of shoulders or hips
         %pred_pose_3d{k}(:,2) = mean(pred_pose_3d{k}(:,[3,6]),2);
         %pred_pose_3d{k}(:,15) = mean(pred_pose_3d{k}(:,[9,12]),2);
         %Center the predictions at the pelvis
         if is_relative == 1
             pred_pose_3d{k} = pred_pose_3d{k} - repmat(pred_pose_3d{k}(:,15), 1, 17);
         else
             pred_pose_3d{k} = pred_pose_3d{k};
         end

         %Other mappings that may be needed to convert the predicted pose to match our coordinate system
         %pred_pose_3d{k} = 1000* pred_pose_3d{k}([2 3 1],:);
         %pred_pose_3d{k}(1:2,:) = -pred_pose_3d{k}(1:2,:);
     end
    
    %Match predictions to GT 
    [matching, old_matched] = mpii_multiperson_get_identity_matching(gt_pose_2d, gt_visibility, pred_pose_2d, pred_visibility, 40);
    
    undetected_people = undetected_people + sum(matching == 0);
    
    for k = 1:valid_annotations
        if is_relative == 1
            P = gt_pose_3d{k}(:,1:num_joints) - repmat(gt_pose_3d{k}(:,15),1 , num_joints);
        else
            P = gt_pose_3d{k}(:,1:num_joints);
        end

        pred_considered = 0;
        
        if(matching(k) ~= 0 )
            pred_p = pred_pose_3d{matching(k)}(:,1:num_joints);
            pred_p = mpii_map_to_gt_bone_lengths(pred_p, P, o1, safe_traversal_order(2:end));
            pred_considered = 1;
        else
            pred_p = 100000 * ones(size(P)); %So that the 3DPCK metric marks all these joints as 0!
            if(EVALUATION_MODE==0)
                pred_considered = 1;
            end
        end
        
        if (pred_considered == 1 )
            error_p = (pred_p - P).^2;
            error_p = sqrt(sum(error_p, 1));
            per_joint_error(1:num_joints,1,pje_idx) = error_p;     
            per_joint_occlusion_mask(1:num_joints,1,pje_idx) = gt_pose_occlusion_labels{k};
            per_joint_visibility_mask(1:num_joints,1,pje_idx) = gt_pose_visibility_labels{k};
            pje_idx = pje_idx + 1;
        end
        
    end

end
sequencewise_undetected_people(ts) = undetected_people;
sequencewise_annotated_people(ts) = annotated_people;
sequencewise_per_joint_error{ts} = per_joint_error;
sequencewise_visibility_mask{ts} =  per_joint_visibility_mask;
sequencewise_occlusion_mask{ts} =  per_joint_occlusion_mask;  
    
end


if(EVALUATION_MODE == 0)
    out_prefix = 'all_annotated_';
else
    out_prefix = 'only_matched_annotations_';
end

save([results_output_path filesep out_prefix 'multiperson_3dhp_evaluation.mat'], 'sequencewise_per_joint_error' );

[seq_table] = mpii_evaluate_multiperson_errors(sequencewise_per_joint_error );%fullfile(net_base, net_path{n,1}));
out_file = [results_output_path filesep out_prefix 'multiperson_3dhp_evaluation'];
writetable(cell2table(seq_table), [out_file '_sequencewise.csv']);

  
[seq_table] = mpii_evaluate_multiperson_errors_visibility_mask(sequencewise_per_joint_error , sequencewise_visibility_mask);
out_file = [results_output_path filesep [out_prefix 'visible_joints_'] 'multiperson_3dhp_evaluation'];
writetable(cell2table(seq_table), [out_file '_sequencewise.csv']);

[seq_table] = mpii_evaluate_multiperson_errors_visibility_mask(sequencewise_per_joint_error , sequencewise_occlusion_mask);
out_file = [results_output_path filesep [out_prefix 'occluded_joints_'] 'multiperson_3dhp_evaluation'];
writetable(cell2table(seq_table), [out_file '_sequencewise.csv']);
  
%
end
