function myAffinityMat = Image2Graph(imIn)
  imDims = size(imIn, [1 2]);
  myAffinityMat = zeros(imDims(1)*imDims(2));
  for i = 1:imDims(1)
    for j = 1:imDims(2)
      dists = exp(-sqrt(sum((imIn-imIn(i,j,:)).^2,3)));
      myAffinityMat((i-1)+(j-1)*imDims(1)+1,:) = dists(:)';
    end
  end
end