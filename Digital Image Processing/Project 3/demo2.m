clear

%% Load and normalize image stack
exposureTimes = [1/2500, 1/1000, 1/500, 1/250, 1/125, 1/60, ...
                 1/30, 1/15, 1/8, 1/4, 1/2, 1, 2, 4, 8, 15];
stackBaseName = 'Image1\exposure';
stackSize = 16;
stackFileType = 'jpg';
for i=stackSize:-1:1
  file = sprintf("%s%i.%s", stackBaseName, i, stackFileType);
  %imgStack(:,:,i) = double(rgb2gray(imread(file)))/255.0;
  imgStack(:,:,:,i) = double(imread(file))/255.0;
end

%% Merge all LDR pictures
radianceMap = mergeLDRStack(imgStack, exposureTimes, weightFunctions('photon',0.01,0.99,exposureTimes));
radianceMapNorm = toneMapping(radianceMap, 1);

%% Gamma correction
gamma = 0.8871;
tonedImage = toneMapping(radianceMap, gamma);

%% Display HDR image
close all; figure; %clf;
imshow(tonedImage); title("Gamma="+gamma)

%% Plot pixel values
figure; %clf;
tonedImageGray = rgb2gray(tonedImage);
rows = [230 290 340 400 460 510]';
col = 1335;
hold on;
plot(tonedImageGray(rows,col),'o');
plot([1;6],tonedImageGray(rows([1,6]),col),'b');
plot(radianceMapNorm(rows,col),'x');
plot([1;6],radianceMapNorm(rows([1,6]),col),'r');
legend("Brighness on gamma corrected", "Line on gamma corrected", ...
       "Brighness on non corrected", "Line on non corrected", ...
       'Location','northwest')