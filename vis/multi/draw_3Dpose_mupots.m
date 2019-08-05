function draw_3Dpose_mupots()
 
    root_path = '/mnt/hdd1/Data/Human_pose_estimation/MU/mupots-3d-eval/MultiPersonTestSet/';
    save_path = './vis/';
    num_joint =  17;

    colorList_skeleton = [
    255/255 128/255 0/255;
    255/255 153/255 51/255;
    255/255 178/255 102/255;
    230/255 230/255 0/255;

    255/255 153/255 255/255;
    153/255 204/255 255/255;

    255/255 102/255 255/255;
    255/255 51/255 255/255;

    102/255 178/255 255/255;
    51/255 153/255 255/255;

    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;

    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    ];
    colorList_joint = [
    255/255 128/255 0/255;
    255/255 153/255 51/255;
    255/255 153/255 153/255;
    255/255 102/255 102/255;
    255/255 51/255 51/255;
    153/255 255/255 153/255;
    102/255 255/255 102/255;
    51/255 255/255 51/255;
    255/255 153/255 255/255;
    255/255 102/255 255/255;
    255/255 51/255 255/255;
    153/255 204/255 255/255;
    102/255 178/255 255/255;
    51/255 153/255 255/255;
    230/255 230/255 0/255;
    230/255 230/255 0/255;
    255/255 178/255 102/255;

    ];
    skeleton = [ [0, 16], [1, 16], [1, 15], [15, 14], [14, 8], [14, 11], [8, 9], [9, 10], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7] ];
    skeleton = transpose(reshape(skeleton,[2,16])) + 1;

    fp_img_name = fopen('../mupots_img_name.txt');
    preds_2d_kpt = load('preds_2d_kpt_mupots.mat');
    preds_3d_kpt = load('preds_3d_kpt_mupots.mat');

    img_name = fgetl(fp_img_name);
    while ischar(img_name)
        img_name_split = strsplit(img_name);
        folder_id = str2double(img_name_split(1)); frame_id = str2double(img_name_split(2));
        img_name = sprintf('TS%d/img_%06d.jpg',folder_id, frame_id);
        img_path = strcat(root_path,img_name);

        pred_2d_kpt = getfield(preds_2d_kpt,sprintf('TS%d_img_%06d',folder_id, frame_id));
        pred_3d_kpt = getfield(preds_3d_kpt,sprintf('TS%d_img_%06d',folder_id, frame_id));

        %img = draw_2Dskeleton(img_path,pred_2d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);
        img = imread(img_path);
        f = draw_3Dskeleton(img,pred_3d_kpt,num_joint,skeleton,colorList_joint,colorList_skeleton);

        set(gcf, 'InvertHardCopy', 'off');
        set(gcf,'color','w');
        mkdir(strcat(save_path,sprintf('TS%d',folder_id)));
        saveas(f, strcat(save_path,img_name));
        close(f);

        img_name = fgetl(fp_img_name);
    end
        
end
