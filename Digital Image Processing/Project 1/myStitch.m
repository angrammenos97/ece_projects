function Im=myStitch(im1,im2)
  %% Define local decriptor parameters
  rhom = 10;
  rhoM = 30;
  rhostep = 1;
  N = 16;
  %% Apply interpolation to smooth noise, grayscale and normalize images
  im1_0 = myImgRotation(im1, 0);
  im1_0_gray = rgb2gray(im1_0);
  im1_0norm = double(im1_0_gray)/255.0;
  im2_0 = myImgRotation(im2, 0);
  im2_0_gray = rgb2gray(im2_0);
  im2_0norm = double(im2_0_gray)/255.0;
  %% Find corners using harris corner detection
  corners1 = myDetectHarrisFeatures(im1_0norm);
  corners2 = myDetectHarrisFeatures(im2_0norm);
  %% Find two most similar corners
  %corners = [inf 1 1 ; inf 1 1];
  mapCorners = zeros([size(corners1,1), 2]);
  mapCorners(:,1) = (1:size(corners1,1))';
  bestErrors = inf*ones([size(corners1,1), 1]);
  for i=1:size(corners1,1)
    %d1 = myLocalDescriptor(im1_0_gray, corners1(i,:), rhom, rhoM, rhostep, N);
    d1 = myLocalDescriptorUpgrade(im1_0_gray, corners1(i,:), rhom, rhoM, rhostep, N);
    if isempty(d1)
      continue
    end
    bestErrors(i) = inf;
    for j=1:size(corners2,1)
      %d2 = myLocalDescriptor(im2_0_gray, corners2(j,:), rhom, rhoM, rhostep, N);
      d2 = myLocalDescriptorUpgrade(im2_0_gray, corners2(j,:), rhom, rhoM, rhostep, N);
      if isempty(d2)
        continue
      end
      error = sum(abs(d1-d2));
      if error < bestErrors(i)
        mapCorners(i,2) = j;
        bestErrors(i) = error;
      end
    end
  end
  %% Remove duplicates
  [~, idx] = sort(mapCorners, 1);
  mapCorners = mapCorners(idx(:,2),:);
  bestErrors = bestErrors(idx(:,2));
  for i=1:size(mapCorners,1)
    while (i+1<=size(mapCorners,1)) && (mapCorners(i+1,2)==mapCorners(i,2))
      if bestErrors(i) < bestErrors(i+1)
        mapCorners(i,:) = [];
        bestErrors(i) = [];
      else
        mapCorners(i+1,:) = [];
        bestErrors(i+1) = [];
      end
    end
  end
  %% Select identical points
  %Sort by minimum error and keep the best of them
  [bestErrors, idx] = sort(bestErrors);
  mapCorners = mapCorners(idx,:);
  bestErrorThres = 45;
  bestErrors = bestErrors(bestErrors<bestErrorThres);
  mapCorners = mapCorners(bestErrors<45,:);
  cornersDist = zeros(size(mapCorners, 1));
  farestCorners = [1 2];
  farestDist = 0;
  for i=1:size(cornersDist,1)
    for j=1+i:size(cornersDist,2)
      cornersDist(i,j) =  sqrt( (corners1(mapCorners(i,1),1)-corners1(mapCorners(j,1),1))^2 + ...
                                (corners1(mapCorners(i,1),1)-corners1(mapCorners(j,1),1))^2);
      if cornersDist(i,j)>farestDist
        farestCorners = [i, j];
        farestDist = cornersDist(i,j);
      end
    end
  end
  points1 = corners1(mapCorners(farestCorners,1),:);
  points2 = corners2(mapCorners(farestCorners,2),:);
  %% Find angle difference
  theta1 = atan((points1(2,2)-points1(1,2))/(points1(2,1)-points1(1,1)));
  theta2 = atan((points2(2,2)-points2(1,2))/(points2(2,1)-points2(1,1)));
  thetaDiff = rad2deg(theta1-theta2);
  %% Rotate the second image
  im2_rot = myImgRotation(im2, thetaDiff);
  im2Size = size(im2_0, [1 2]);
  im2RotSize = size(im2_rot, [1 2]);
  points2_rot(:,1) = floor(cosd(thetaDiff)*(points2(:,1)-im2Size(1)/2)-sind(thetaDiff)*(points2(:,2)-im2Size(2)/2)+im2RotSize(1)/2);
  points2_rot(:,2) = floor(+sind(thetaDiff)*(points2(:,1)-im2Size(1)/2)+cosd(thetaDiff)*(points2(:,2)-im2Size(2)/2)+im2RotSize(2)/2);
  %% Merge two images
  im1Size = size(im1_0, [1 2]);
  firstOffTop = max(0, points2_rot(1,1)-points1(1,1));
  firstOffLeft =  max(0, points2_rot(1,2)-points1(1,2));  
  mergedSize = [firstOffTop+points1(1,1)+(im2RotSize(1)-points2_rot(1,1)), ...
                firstOffLeft+points1(1,2)+(im2RotSize(2)-points2_rot(1,2)), 3];
  Im = zeros(mergedSize, 'uint8');
  Im(1+firstOffTop:im1Size(1)+firstOffTop,1+firstOffLeft:im1Size(2)+firstOffLeft,:) = im1_0;
  for i=1:im2RotSize(1)
    for j=1:im2RotSize(2)
      if any(im2_rot(i,j,:))
        Im(points1(1,1)-points2_rot(1,1)+i,points1(1,2)-points2_rot(1,2)+j,:) = im2_rot(i,j,:);
      end
    end
  end
end