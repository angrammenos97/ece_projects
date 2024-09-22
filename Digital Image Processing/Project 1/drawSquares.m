function boxedImg = drawSquares(img, points, squareSize)
  imgSize = size(img);
  if size(imgSize, 2) == 2
    boxedImg = zeros([imgSize(1:2) 3]); %expand dimention
    boxedImg(:,:,1) = img;
    boxedImg(:,:,2) = img;
    boxedImg(:,:,3) = img;
  else
    boxedImg = img;
  end
  squareCenter = floor((squareSize+1)/2);
  for p=1:size(points,1)
    p1 = points(p,1);
    p2 =  points(p,2);
    for i=1:squareSize
      for j=1:squareSize
        boxedImg(p1+i-squareCenter,p2+j-squareCenter,:) = [255,0,0];
      end
    end
  end
end