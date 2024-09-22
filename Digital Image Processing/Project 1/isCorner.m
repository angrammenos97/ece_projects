function [c, R] = isCorner(I, p, k, Rthres)
  %% Initialize
  iSize = size(I);
  regionWindow = 5;
  centerOfRegion = floor((regionWindow+1)/2);
  I1 = zeros(regionWindow);
  I2 = zeros(regionWindow);
  mask1 = [1, -1];  %flipped convolutional mask for dim 1
  mask2 = [1; -1];  %flipped convolutional mask for dim 2
  p1 = p(1);
  p2 = p(2);
  M = zeros(2);
  c = false;
  R = 0;
  %% Define circular window
  sigma = sqrt(floor(regionWindow));
  w = @(x1,x2) exp(-(x1^2+x2^2)/(sigma^2));
  %% Calculate M matrix
  if  p1 - (centerOfRegion-1) > 0 && p1 + (centerOfRegion-1) < iSize(1) && ...
      p2 - (centerOfRegion-1) > 0 && p2 + (centerOfRegion-1) < iSize(2)
    for i=1:regionWindow
      for j=1:regionWindow
        u1 = i-centerOfRegion;
        u2 = j-centerOfRegion;
        %Calculate I1 and I2
        I1(i,j) = mask1(1)*I(p1+u1,p2+u2) + mask1(2)*I(p1+u1,p2+u2+1);
        I2(i,j) = mask2(1)*I(p1+u1,p2+u2) + mask2(2)*I(p1+u1+1,p2+u2);
        %Calculate A matrix
        A = [ I1(i,j)^2,        I1(i,j)*I2(i,j); ...
              I1(i,j)*I2(i,j),  I2(i,j)^2];
        M = M + w(u1,u2)*A;
      end
    end
  else  %out of image borders
    return; %it's not a corner
  end
  %% Calculate R
  R = det(M)-k*trace(M)^2;
  %% Decide if it is corner
  if R > Rthres
    c = true; %it's corner
  end
end