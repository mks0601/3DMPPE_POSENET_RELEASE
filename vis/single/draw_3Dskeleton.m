function f = draw_3Dskeleton(pred_3d_kpt, num_joint, skeleton, colorList_joint, colorList_skeleton)
    
    pred_3d_kpt = squeeze(pred_3d_kpt);

    x = pred_3d_kpt(:,1);
    y = pred_3d_kpt(:,2);
    z = pred_3d_kpt(:,3);
    pred_3d_kpt(:,1) = -z;
    pred_3d_kpt(:,2) = x;
    pred_3d_kpt(:,3) = -y;

    
    f = figure;%('Position',[100 100 600 600]);
    set(f, 'visible', 'off');
    hold on;
    grid on;
    line_width = 6;
 
    num_skeleton = size(skeleton,1);
    for j =1:num_skeleton
        k1 = skeleton(j,1);
        k2 = skeleton(j,2);

        plot3([pred_3d_kpt(k1,1),pred_3d_kpt(k2,1)],[pred_3d_kpt(k1,2),pred_3d_kpt(k2,2)],[pred_3d_kpt(k1,3),pred_3d_kpt(k2,3)],'Color',colorList_skeleton(j,:),'LineWidth',line_width);
    end
    for j=1:num_joint
        scatter3(pred_3d_kpt(j,1),pred_3d_kpt(j,2),pred_3d_kpt(j,3),100,colorList_joint(j,:),'filled');
    end
   
    set(gca, 'color', [255/255 255/255 255/255]);
    set(gca,'XTickLabel',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'ZTickLabel',[]);
    
    x = pred_3d_kpt(:,1);
    xmin = min(x(:)) - 100;
    xmax = max(x(:)) + 100;
    
    y = pred_3d_kpt(:,2);
    ymin = min(y(:)) - 100;
    ymax = max(y(:)) + 100;

    z = pred_3d_kpt(:,3);
    zmin = min(z(:));
    zmax = max(z(:)) + 100;

    xcenter = mean(pred_3d_kpt(:,1));
    ycenter = mean(pred_3d_kpt(:,2));
    zcenter = mean(pred_3d_kpt(:,3));
    xmin = xcenter - 1000;
    xmax = xcenter + 1000;
    ymin = ycenter - 1000;
    ymax = ycenter + 1000;
    zmin = zcenter - 1000;
    zmax = zcenter + 1000;
    
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    zlim([zmin zmax]);
    
    view(62,7);
end
