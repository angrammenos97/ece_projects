function d = myLocalDescriptorUpgrade(I,p,rhom,rhoM,rhostep ,N)
  %% Initialize
  imgSize = size(I);
  rho = rhoM:-rhostep:rhom;
  d = zeros(size(rho));
  %% Perform rotation invariant
  for r=rho %for every radius
    xrho = zeros(size(1:N));
    for n=1:N %for every point on circle
      %Find the angle of the point on the circle
      theta = (n*2*pi/N)+(pi/2);
      %Find its x and y coordinates
      x = r*cos(theta);
      y = r*sin(theta);
      %Get fractional and decimal value to calculate interpolation
      x_fraction = floor(x);
      %x_decimal = x-x_fraction;
      y_fraction = floor(y);
      %y_decimal = y - y_fraction;
      if  p(1)+x_fraction>=2 && p(1)+x_fraction<=imgSize(1)-2 && ...
          p(2)+y_fraction>=2 && p(2)+y_fraction<=imgSize(2)-2 %if not image borders exceeded
         %Find the 16 pixels around the point and their distances
         minNeighborsDist = inf*ones([4,1]);
         minNeighborsIdxs = zeros([4,2]);
         for i=-1:2
          for j=-1:2
            dist = abs(((x_fraction+i)^2 + (y_fraction+j)^2)-r^2);
            if max(minNeighborsDist) > dist
              minNeighborsDist = [minNeighborsDist; dist];
              minNeighborsIdxs = [minNeighborsIdxs; i j];
              [minNeighborsDist, idx] = sort(minNeighborsDist);
              minNeighborsIdxs = minNeighborsIdxs(idx, :);
              minNeighborsDist = minNeighborsDist(1:end-1);
              minNeighborsIdxs = minNeighborsIdxs(1:end-1,:);
            end
          end
         end
         xrho(n) =mean([I(p(1)+x_fraction+minNeighborsIdxs(1,1),p(2)+y_fraction+minNeighborsIdxs(1,2)) ...
                        I(p(1)+x_fraction+minNeighborsIdxs(2,1),p(2)+y_fraction+minNeighborsIdxs(2,2)) ...
                        I(p(1)+x_fraction+minNeighborsIdxs(3,1),p(2)+y_fraction+minNeighborsIdxs(3,2)) ...
                        I(p(1)+x_fraction+minNeighborsIdxs(4,1),p(2)+y_fraction+minNeighborsIdxs(4,2))]);

%         x_indices = [x_fraction-1, x_fraction, x_fraction+1, x_fraction+2];
%         y_indices = [y_fraction-1, y_fraction, y_fraction+1, y_fraction+2];
%         minNeighborsDist = min(imgSize)*ones([4,4]);
%         for i=1:4
%           for j=1:4
%             minNeighborsDist(i,j) = abs(sqrt(x_indices(j)^2+y_indices(i)^2)-r);
%           end
%         end
%         %Find the 4 pixels closest to the circle
%         [~, minNeighborsIndices] = sort(minNeighborsDist(:),'ascend');
%         minNeighborsIndices = minNeighborsIndices(1:4);
%         j_Neighbors = rem(minNeighborsIndices-1,4)+1;
%         i_Neighbors = floor((minNeighborsIndices-1)/4)+1;
%         %Get their mean value
%         xrho(n) = mean([I(p(1)+y_fraction+i_Neighbors(1),p(2)+x_fraction+j_Neighbors(1)) ...
%                         I(p(1)+y_fraction+i_Neighbors(2),p(2)+x_fraction+j_Neighbors(2)) ...
%                         I(p(1)+y_fraction+i_Neighbors(3),p(2)+x_fraction+j_Neighbors(3)) ...
%                         I(p(1)+y_fraction+i_Neighbors(4),p(2)+x_fraction+j_Neighbors(4))]);
      else  %out of image borders
        %Return empty vector
        d = [];
        return
      end
    end
    %Each element of the returned vector is the mean of all values of the
    %points on the circle
    d(r-rhom+1) = mean(xrho);
  end
end