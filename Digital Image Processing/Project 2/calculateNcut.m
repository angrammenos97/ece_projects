function nCutValue = calculateNcut(anAffinityMat , clusterIdx)
  idxA = clusterIdx==1;
  idxB = clusterIdx==2;
  assocAA = sum(anAffinityMat(idxA,idxA),"all");
  assocAV = sum(anAffinityMat(idxA,:),"all");
  assocBB = sum(anAffinityMat(idxB,idxB),"all");
  assocBV = sum(anAffinityMat(idxB,:),"all");
  Nassoc = (assocAA/assocAV)+(assocBB/assocBV);
  nCutValue = 2 - Nassoc;
end