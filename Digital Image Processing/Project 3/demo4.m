clear

%% Load image stack
stackBaseName = 'Image2\sample2-0';
stackSize = 7;
stackFileType = 'jpg';
for i=stackSize:-1:1
  file = sprintf("%s%i.%s", stackBaseName, i-1, stackFileType);
  imgStack(:,:,:,i) = imread(file);
end
exposureTimes = [1/400, 1/250, 1/100, 1/40, 1/25, 1/8, 1/3];

%% Merge all LDR pictures
radianceMap = mergeLDRStack(double(imgStack)/255.0, exposureTimes, weightFunctions('photon',0.01,0.99,exposureTimes));

%% Gamma correction
gamma = 1;
tonedImage = toneMapping(radianceMap, gamma);

%% Allign images, crop and normalize
[optimizer, metric] = imregconfig("monomodal");
imgStack(:,:,1,6) = imregister(imgStack(:,:,1,6), imgStack(:,:,1,5),'rigid',optimizer,metric);
imgStack(:,:,2,6) = imregister(imgStack(:,:,2,6), imgStack(:,:,2,5),'rigid',optimizer,metric);
imgStack(:,:,3,6) = imregister(imgStack(:,:,3,6), imgStack(:,:,3,5),'rigid',optimizer,metric);
imgStack = double(imgStack(15:709,10:1078,:,:))/255.0;

%% Merge all LDR pictures
radianceMapCorrected = mergeLDRStack(imgStack, exposureTimes, weightFunctions('photon',0.01,0.99,exposureTimes));

%% Gamma correction
gammaCorrected = 1;
tonedImageCorrected = toneMapping(radianceMapCorrected, gammaCorrected);

%% Display HDR image
%close all; figure;
tiledlayout(2,1,"TileSpacing",'tight');
nexttile; imshow(tonedImage); title("No correction, Gamma="+gamma)
nexttile; imshow(tonedImageCorrected); title("With correction, Gamma="+gammaCorrected)


