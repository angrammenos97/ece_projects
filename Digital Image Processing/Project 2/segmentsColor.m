function outputImage = segmentsColor(imIn, labels)
  outputImage = zeros(size(imIn), class(imIn));
  if size(imIn,3) == 1  %grayscale
     for l=0:max(labels(:))
      outputImage(labels==l) = mean(imIn(labels==l));
    end
  else                  %rgb
    imInR = imIn(:,:,1);
    imInG = imIn(:,:,2);
    imInB = imIn(:,:,3);
    outputImageR = zeros(size(imInR), class(imIn));
    outputImageG = zeros(size(imInG), class(imIn));
    outputImageB = zeros(size(imInB), class(imIn));
    for l = unique(labels)'
      outputImageR(labels==l) = mean(imInR(labels==l));
      outputImageG(labels==l) = mean(imInG(labels==l));
      outputImageB(labels==l) = mean(imInB(labels==l));
    end
    outputImage(:,:,1) = outputImageR;
    outputImage(:,:,2) = outputImageG;
    outputImage(:,:,3) = outputImageB;
  end
end