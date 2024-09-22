clear
img = imread("bee.jpg");

[superpixelLabels,~] = slicmex(img, 400, 20);
imgNorm = double(img)/255.0;
superpixelDescriptions = superpixelDescriptor(imgNorm, superpixelLabels);
superpixelLabelsU = unique(superpixelLabels)';
superpixelDescriptionsU = zeros([size(superpixelLabelsU) size(superpixelDescriptions,3)]);
for l = 1:size(superpixelLabelsU,2)
  [rl,cl] = find(superpixelLabels==superpixelLabelsU(l),1);
  superpixelDescriptionsU(1,l,:) = superpixelDescriptions(rl,cl,:);
end
affinityMat = Image2Graph(superpixelDescriptionsU);

%Non-recursive algorithms
labels_k6 = myGraphSpectralClustering(affinityMat, 6);    %k=6
finalLabels_k6 = labels_k6(superpixelLabels+1);
labels_k10 = myGraphSpectralClustering(affinityMat, 10);  %k=10
finalLabels_k10 = labels_k10(superpixelLabels+1);

%Recursive algorithm
T1 = 21;
T2 = 0.9832;
nCutLabels = ones(size(affinityMat,1),1);
[nCutLabels,clustersNum]=nCuts(affinityMat, nCutLabels, T1, T2);
finalLabels = nCutLabels(superpixelLabels+1);

%close all; 
tiledlayout(2,2);
nexttile;imshow(segmentsColor(img, finalLabels_k6)); title("Non,k=6")
nexttile;imshow(segmentsColor(img, finalLabels_k10)); title("Non,k=10")
nexttile;imshow(segmentsColor(img, finalLabels)); title("k="+clustersNum)
nexttile;imshow(superpixelDescriptions); title("Original")
