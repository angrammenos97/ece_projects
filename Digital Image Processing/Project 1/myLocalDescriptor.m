function d = myLocalDescriptor(I,p,rhom,rhoM,rhostep ,N)
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
      x_decimal = x-x_fraction;
      y_fraction = floor(y);
      y_decimal = y - y_fraction;
      if  p(1)+x_fraction>=1 && p(1)+x_fraction<imgSize(1) && ...
          p(2)+y_fraction>=1 && p(2)+y_fraction<imgSize(2) %if not image borders exceeded
        %Calculate interpolated value of four nearest pixels
        xrho(n) = I(p(1)+x_fraction,p(2)+y_fraction)*(1-x_decimal)*(1-y_decimal) +...
                  I(p(1)+x_fraction+1,p(2)+y_fraction)*x_decimal*(1-y_decimal) +...
                  I(p(1)+x_fraction,p(2)+y_fraction+1)*(1-x_decimal)*y_decimal +...
                  I(p(1)+x_fraction+1,p(2)+y_fraction+1)*x_decimal*y_decimal;
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
