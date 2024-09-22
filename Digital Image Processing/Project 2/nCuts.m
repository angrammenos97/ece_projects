function [labels,clustersNum]=nCuts(anAffinityMat, labels, T1, T2)
  clustersNum = labels(1);
  if size(labels)==1  %can't segment graph with one segment
    return
  end
  tmp_labels = myGraphSpectralClustering(anAffinityMat, 2);
  if  (size(find(tmp_labels==1),1)>=T1) && (size(find(tmp_labels==2),1)>=T1)
    nCutValue = calculateNcut(anAffinityMat, tmp_labels);
    if nCutValue<=T2
      %Recursive call for nodes from segment 1
      [labels(tmp_labels==1),clustersNum] = nCuts(anAffinityMat(tmp_labels==1,tmp_labels==1), ...
                                                    labels(tmp_labels==1), T1, T2);
      %Recursive call for nodes from segment 2
      labels(tmp_labels==2)=clustersNum+1;
      [labels(tmp_labels==2),clustersNum] = nCuts(anAffinityMat(tmp_labels==2,tmp_labels==2), ...
                                                    labels(tmp_labels==2), T1, T2);
    end
  end
end