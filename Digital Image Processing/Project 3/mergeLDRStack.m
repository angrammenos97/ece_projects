function radianceMap = mergeLDRStack(imgStack, exposureTimes , weightingFcn)
  %% Define min and max values
  Zmin = 0.01;
  Zmax = 0.99;  
  %% Fix for grayscale
  if size(size(imgStack),2)==3
    imgSize = size(imgStack,[1,2]);
    stackSize = size(imgStack, 3);
    imgStack = reshape(imgStack, [imgSize,1,stackSize]);
  end
  %% Calculate ln(Eij)
  imgSize = size(imgStack, [1,2,3]);
  stackSize = size(imgStack, 4);
  for i=imgSize(1):-1:1
    for j=imgSize(2):-1:1
      Zij = reshape(imgStack(i,j,:,:), [imgSize(3),stackSize]);
      underExposured(i,j,:) = all(Zij<Zmin,2);
      overExposured(i,j,:) = all(Zij>Zmax,2);        
      wZij = weightingFcn(Zij);
      sumwZij = sum(wZij,2);
      Zij(Zij==0) = Zmin; %fix zero values causing NaN
      lnE(i,j,:) = (sum(wZij.*(log(Zij)-log(exposureTimes)),2))./sumwZij;
    end
  end
  %% Fix underexposured and overexposured pixels
  lnE(isnan(lnE)) = 0.0;
  lnE = lnE + (min(lnE,[],[1,2]).*underExposured);
  lnE = lnE + (max(lnE,[],[1,2]).*overExposured);
  %% Return exposure
  %radianceMap = exp(lnE);
  radianceMap = lnE;
end