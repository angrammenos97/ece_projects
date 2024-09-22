function rotImg = myImgRotation(img, angle)
  %% Apply interpolation in original image
  H = [ 0 0.25 0; 0.25 0 0.25; 0 0.25 0];
  img =  convn(img, H);
  img = img(3:end-2,3:end-2,:);
  %% Calculate rotated image size
  imgSize = size(img, [1 2]);
  firstDim = [0 imgSize(1) imgSize(1) 0];
  secondDim = [0 0 imgSize(2) imgSize(2)];
  cosa = cosd(angle);
  sina = sind(angle);
  firstDimRot = cosa*(firstDim-imgSize(1)/2)+sina*(secondDim-imgSize(2)/2);
  secondDimRot = -sina*(firstDim-imgSize(1)/2)+cosa*(secondDim-imgSize(2)/2);
  imgSizeRot = floor([max(firstDimRot-min(min(firstDimRot),1)) ...
                      max(secondDimRot-min(min(secondDimRot),1))]);
  %% Calulate origin pixels
  firstDimIdxRot = repmat((1:imgSizeRot(1))',1,imgSizeRot(2));
  secondDimIdxRot = repmat((1:imgSizeRot(2)),imgSizeRot(1),1);
  firstDimIdx = floor(cosa*(firstDimIdxRot - imgSizeRot(1)/2) + ...
                      sina*(secondDimIdxRot -imgSizeRot(2)/2) + imgSize(1)/2);
  secondDimIdx = floor(-sina*(firstDimIdxRot - imgSizeRot(1)/2) ...
                       +cosa*(secondDimIdxRot -imgSizeRot(2)/2) + imgSize(2)/2);
  % Create the rotated image
  rotImg = zeros([imgSizeRot size(img,3)], 'uint8');
  for i=1:imgSizeRot(1)
    for j=1:imgSizeRot(2)
      if (firstDimIdx(i,j) >=1)&&(firstDimIdx(i,j) <=imgSize(1))&& ...
         (secondDimIdx(i,j)>=1)&&(secondDimIdx(i,j)<=imgSize(2))
        rotImg(i,j,:) = img(firstDimIdx(i,j),secondDimIdx(i,j),:);
      end
    end
  end
end