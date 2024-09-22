%NOTE: execute once the demo2.m to load workspace
gammas = 0.8:0.01:1.4; 
rows = [230 290 340 400 460 510]';
col = 1335;

leastError = inf;
bestGamma = 1;
for gamma=gammas
  tonedImage = double(toneMapping(radianceMap, gamma));
  %tonedImage = double(radianceMapNorm);
  MSE = 0.0;
  v1 = [1, tonedImage(rows(1),col)];
  v2 = [6, tonedImage(rows(6),col)];
  pnt = [(1:6)', tonedImage(rows,col)];
  for i=1:6
    MSE = MSE + distFromLine(v1, v2, pnt(i,:))^2;
  end
  sqrtError = MSE/6;
  if sqrtError<leastError
    leastError = sqrtError;
    bestGamma = gamma;
  end
end
disp("Best gamma = " + bestGamma)

function dist=distFromLine(v1, v2, pnt)
  A = (v2(2)-v1(2))/(v2(1)-v1(1));
  B = -1;
  C = v1(2)-v1(1);
  dist = abs(A*pnt(1)+B*pnt(2)+C)/sqrt((A^2)+(B^2));
end
