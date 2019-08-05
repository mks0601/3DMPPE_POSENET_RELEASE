function img = draw_2Dskeleton(img_name, pred_2d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton)
 
    img = imread(img_name);
    [imgHeight, imgWidth, dim] = size(image);

    f = figure;
    set(f, 'visible', 'off');
    imshow(img);
    hold on;
    line_width = 4;
    
    num_skeleton = size(skeleton,1);

    num_pred = size(pred_2d_kpt,1);
    for i = 1:num_pred
        for j =1:num_skeleton
            k1 = skeleton(j,1);
            k2 = skeleton(j,2);
            plot([pred_2d_kpt(i,k1,1),pred_2d_kpt(i,k2,1)],[pred_2d_kpt(i,k1,2),pred_2d_kpt(i,k2,2)],'Color',colorList_skeleton(j,:),'LineWidth',line_width);
        end
        for j=1:num_joint
            scatter(pred_2d_kpt(i,j,1),pred_2d_kpt(i,j,2),100,colorList_joint(j,:),'filled');
        end
    end
    
    set(gca,'Units','normalized','Position',[0 0 1 1]);  %# Modify axes size

    frame = getframe(gcf);
    img = frame.cdata;
    
    hold off;
    close(f); 

end
