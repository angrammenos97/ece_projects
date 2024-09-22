function corners = myDetectHarrisFeatures(I)
  %% Initialize
  iSize = size(I);
  k = 0.04;
  Rthes = 0.013;
  posibleCorners = zeros(iSize, 'logical');
  Rs = zeros(iSize);
  %% Find corners
  for p1=1:iSize(1)
    for p2=1:iSize(2)
      [posibleCorners(p1,p2), Rs(p1,p2)] = isCorner(I, [p1,p2], k, Rthes);
    end
  end
  %% Refine corners
  corners = [];
  for p1=1:iSize(1)
    for p2=1:iSize(2)
      if posibleCorners(p1,p2)
        i = 0;
        j = 0;
        maxR = 0; %Rs(p1,p2);
        maxi = 0;
        maxj = 0;
        %Search near posible corners
        while (posibleCorners(p1+i,p2) && p1+i<=iSize(1))
          while(posibleCorners(p1+i, p2+j) && p2+j<=iSize(2))
            if Rs(p1+i,p2+j) > maxR %near pixel has greater R
              posibleCorners(p1+maxi,p2+maxj) = false; %previous pixel is not corner
              posibleCorners(p1+i,p2+j) = true;  %this pixel maybe is a corner
              maxR = Rs(p1+i,p2+j);
              maxi = i;
              maxj = j;
            else
              posibleCorners(p1+i,p2+j) = false; %it's not a corner
            end
            j = j+1;
          end
          i = i+1;
        end
        %Save corner of the region
        corners = [corners; p1+maxi p2+maxj];
      end
    end
  end
end