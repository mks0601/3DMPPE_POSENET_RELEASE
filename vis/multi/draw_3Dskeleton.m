function f = draw_3Dskeleton(img, pred_3d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton)
 
    x = pred_3d_kpt(:,:,1);
    y = pred_3d_kpt(:,:,2);
    z = pred_3d_kpt(:,:,3);
    pred_3d_kpt(:,:,1) = -z;
    pred_3d_kpt(:,:,2) = x;
    pred_3d_kpt(:,:,3) = -y;

    [imgHeight, imgWidth, dim] = size(img);
    
    figure_height = 450;
    figure_width = figure_height / imgHeight * imgWidth;
    f = figure('Position',[100 100 figure_width figure_height]);
    set(f, 'visible', 'off');
    hold on;
    grid on;
    line_width = 4;
    point_width = 50;
 
    num_skeleton = size(skeleton,1);

    num_pred = size(pred_3d_kpt,1);
    for i = 1:num_pred
        for j =1:num_skeleton
            k1 = skeleton(j,1);
            k2 = skeleton(j,2);

            plot3([pred_3d_kpt(i,k1,1),pred_3d_kpt(i,k2,1)],[pred_3d_kpt(i,k1,2),pred_3d_kpt(i,k2,2)],[pred_3d_kpt(i,k1,3),pred_3d_kpt(i,k2,3)],'Color',colorList_skeleton(j,:),'LineWidth',line_width);
        end
        for j=1:num_joint
            scatter3(pred_3d_kpt(i,j,1),pred_3d_kpt(i,j,2),pred_3d_kpt(i,j,3),point_width,colorList_joint(j,:),'filled');
        end
    end
   
    set(gca, 'color', [255/255 255/255 255/255]);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'ZTickLabel',[]);
    
    x = pred_3d_kpt(:,:,1);
    xmin = min(x(:)) - 120000;
    xmax = max(x(:)) + 6000;
    
    y = pred_3d_kpt(:,:,2);
    ymin = min(y(:));
    ymax = max(y(:));

    z = pred_3d_kpt(:,:,3);
    zmin = min(z(:));
    zmax = max(z(:));
    
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    zlim([zmin zmax]);
    
    h_img = surf([xmin;xmin],[ymin ymax;ymin ymax],[zmax zmax;zmin zmin],'CData',img,'FaceColor','texturemap');
    set(h_img);
    
    view(62,27);
end
