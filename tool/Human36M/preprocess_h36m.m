% Preprocess human3.6m dataset
% Place this file to the Release-v1.1 folder and run it

function preprocess_h36m()

    close all;
    %clear;
    %clc;

    addpaths;

    %--------------------------------------------------------------------------
    % PARAMETERS

    % Subject (1, 5, 6, 7, 8, 9, 11)
    SUBJECT = [1 5 6 7 8 9 11];
     
    % Action (2 ~ 16)
    ACTION = 2:16;
    
    % Subaction (1 ~ 2)
    SUBACTION = 1:2;
    
    % Camera (1 ~ 4)
    CAMERA = 1:4;
    
    num_joint = 17;
    root_dir = '.'; % define path here
    
    % if rgb sequence is declared in the loop, it causes stuck (do not know
    % reason)
    rgb_sequence = cell(1,100000000);
    COUNT = 1;
    %--------------------------------------------------------------------------
    % MAIN LOOP
    % For each subject, action, subaction, and camera..
    for subject = SUBJECT
        for action = ACTION
            for subaction = SUBACTION
                for camera = CAMERA

                    fprintf('Processing subject %d, action %d, subaction %d, camera %d..\n', ...
                        subject, action, subaction, camera);

                    img_save_dir = sprintf('%s/images/s_%02d_act_%02d_subact_%02d_ca_%02d', ...
                        root_dir, subject, action, subaction, camera);
                    if ~exist(img_save_dir, 'dir')
                        mkdir(img_save_dir);
                    end

                    mask_save_dir = sprintf('%s/masks/s_%02d_act_%02d_subact_%02d_ca_%02d', ...
                        root_dir, subject, action, subaction, camera);
                    if ~exist(mask_save_dir, 'dir')
                        mkdir(mask_save_dir);
                    end

                    annot_save_dir = sprintf('%s/annotations/s_%02d_act_%02d_subact_%02d_ca_%02d', ...
                        root_dir, subject, action, subaction, camera);
                    if ~exist(annot_save_dir, 'dir')
                        mkdir(annot_save_dir);
                    end

                    if (subject==11) && (action==2) && (subaction==2) && (camera==1)
                        fprintf('There is an error in subject 11, action 2, subaction 2, and camera 1\n');
                        continue;
                    end
                    
                    % Select sequence
                    Sequence = H36MSequence(subject, action, subaction, camera);

                    % Get 3D pose and 2D pose
                    Features{1} = H36MPose3DPositionsFeature(); % 3D world coordinates
                    Features{1}.Part = 'body'; % Only consider 17 joints
                    Features{2} = H36MPose3DPositionsFeature('Monocular', true); % 3D camera coordinates
                    Features{2}.Part = 'body'; % Only consider 17 joints
                    Features{3} = H36MPose2DPositionsFeature(); % 2D image coordinates
                    Features{3}.Part = 'body'; % Only consider 17 joints
                    F = H36MComputeFeatures(Sequence, Features);
                    num_frame = Sequence.NumFrames;
                    pose3d_world = reshape(F{1}, num_frame, 3, num_joint);
                    pose3d = reshape(F{2}, num_frame, 3, num_joint);
                    pose2d = reshape(F{3}, num_frame, 2, num_joint);

                    % Camera (in global coordinate)
                    Camera = Sequence.getCamera();

                    % Sanity check
                    if false
                        R = Camera.R; % rotation matrix
                        T = Camera.T'; % origin of the world coord system
                        K = [Camera.f(1)    0           Camera.c(1);
                            0              Camera.f(2) Camera.c(2);
                            0              0           1]; % f: focal length, c: principal points
                        error = 0;
                        for i = 1:num_frame
                            X = squeeze(pose3d_global(i,:,:));
                            x = squeeze(pose2d(i,:,:));
                            px = K*R*(X-T);
                            px = px ./ px(3,:);
                            px = px(1:2,:);
                            error = error + mean(sqrt(sum((px-x).^2, 1)));
                        end
                        error = error / num_frame;
                        fprintf('reprojection error = %.2f (pixels)\n', error);
                        keyboard;
                    end

                    %% Image, bounding box for each sampled frame
                    fprintf('Load RGB video: ');
                    rgb_extractor = H36MRGBVideoFeature();
                    rgb_sequence{COUNT} = rgb_extractor.serializer(Sequence);
                    fprintf('Done!!\n');
                    img_height = zeros(num_frame,1);
                    img_width = zeros(num_frame,1);

                    fprintf('Load mask video: ');
                    mask_extractor = H36MMyBGMask();
                    mask_sequence = mask_extractor.serializer(Sequence);
                    fprintf('Done!!\n');


               
                    % For each frame,
                    for i = 1:num_frame
                        if mod(i,100) == 1
                            fprintf('.');
                        end
                       
                        % Save image
                        % Get data
                        img = rgb_sequence{COUNT}.getFrame(i);  
                        [h, w, c] = size(img);
                        img_height(i) = h;
                        img_width(i) = w;
                        img_name = sprintf('%s/s_%02d_act_%02d_subact_%02d_ca_%02d_%06d.jpg', ...
                            img_save_dir, subject, action, subaction, camera, i);
                        %imwrite(img, img_name);

                        mask = mask_sequence.Buffer{i};
                        mask_name = sprintf('%s/s_%02d_act_%02d_subact_%02d_ca_%02d_%06d.jpg', ...
                            mask_save_dir, subject, action, subaction, camera, i);
                        imwrite(mask, mask_name);
                        
                    end
                    
                    COUNT = COUNT + 1;
                    
                    % Save data
                    pose3d_world = permute(pose3d_world,[1,3,2]); % world coordinate 3D keypoint coordinates
                    R = Camera.R; % rotation matrix
                    T = Camera.T; % origin of the world coord system
                    f = Camera.f; % focal length
                    c = Camera.c; % principal points
                    filename = sprintf('%s/h36m_meta.mat', annot_save_dir);
                    %save(filename, 'pose3d_world', 'f', 'c', 'R', 'T', 'img_height', 'img_width');
                    
                    fprintf('\n');
                    
                end
            end
        end
    end

end

