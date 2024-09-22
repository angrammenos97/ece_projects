function clusterIdx = myGraphSpectralClustering(anAffinityMat , k)
  W = anAffinityMat;
  sizeW = size(W);
  D = zeros(size(W));
  for i = 1:sizeW(1)
    D(i,i) = sum(W(i,:));
  end

  L = D-W;
  [U,~] = eigs(L, D, k,'smallestreal');
  
  clusterIdx = kmeans(U, k);
end