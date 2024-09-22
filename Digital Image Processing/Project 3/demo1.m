clear

%% Load and normalize image stack
stackBaseName = 'Image1\exposure';
stackSize = 16;
stackFileType = 'jpg';
for i=stackSize:-1:1
  file = sprintf("%s%i.%s", stackBaseName, i, stackFileType);
  imgStack(:,:,:,i) = double(imread(file))/255.0;
end
exposureTimes = [1/2500, 1/1000, 1/500, 1/250, 1/125, 1/60, ...
                 1/30, 1/15, 1/8, 1/4, 1/2, 1, 2, 4, 8, 15];

%% Merge all LDR pictures
radianceMapUni = mergeLDRStack(imgStack, exposureTimes, weightFunctions('uniform',0.01,0.99));   disp("Done Uniform");
radianceMapTen = mergeLDRStack(imgStack, exposureTimes, weightFunctions('tent',0.01,0.99));      disp("Done Tent");
radianceMapGau = mergeLDRStack(imgStack, exposureTimes, weightFunctions('gaussian',0.01,0.99));  disp("Done Gaussian");
radianceMapPho = mergeLDRStack(imgStack, exposureTimes, weightFunctions('photon',0.01,0.99,exposureTimes));    disp("Done Photon");

%% Display results
close all; figure("Name","Red channel"); tiledlayout(4,2);
nexttile; imagesc(radianceMapUni(:,:,1)); title("Uniform");  colorbar;
nexttile; imagesc(radianceMapTen(:,:,1)); title("Tent");     colorbar;
nexttile; histogram(radianceMapUni(:,:,1)); nexttile; histogram(radianceMapTen(:,:,1));
nexttile; imagesc(radianceMapGau(:,:,1)); title("Gaussian"); colorbar;
nexttile; imagesc(radianceMapPho(:,:,1)); title("Photon");   colorbar;
nexttile; histogram(radianceMapGau(:,:,1)); nexttile; histogram(radianceMapTen(:,:,1));

figure("Name","Green channel"); tiledlayout(4,2);
nexttile; imagesc(radianceMapUni(:,:,2)); title("Uniform");  colorbar;
nexttile; imagesc(radianceMapTen(:,:,2)); title("Tent");     colorbar;
nexttile; histogram(radianceMapUni(:,:,2)); nexttile; histogram(radianceMapTen(:,:,2));
nexttile; imagesc(radianceMapGau(:,:,2)); title("Gaussian"); colorbar;
nexttile; imagesc(radianceMapPho(:,:,2)); title("Photon");   colorbar;
nexttile; histogram(radianceMapGau(:,:,2)); nexttile; histogram(radianceMapPho(:,:,2));

figure("Name","Blue channel"); tiledlayout(4,2);
nexttile; imagesc(radianceMapUni(:,:,3)); title("Uniform");  colorbar;
nexttile; imagesc(radianceMapTen(:,:,3)); title("Tent");     colorbar;
nexttile; histogram(radianceMapUni(:,:,3)); nexttile; histogram(radianceMapTen(:,:,3));
nexttile; imagesc(radianceMapGau(:,:,3)); title("Gaussian"); colorbar;
nexttile; imagesc(radianceMapPho(:,:,3)); title("Photon");   colorbar;
nexttile; histogram(radianceMapGau(:,:,3)); nexttile; histogram(radianceMapPho(:,:,3));