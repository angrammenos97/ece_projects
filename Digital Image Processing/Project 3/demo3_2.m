clear

%% Load image stack
stackBaseName = 'Image2\sample2-0';
stackSize = 7;
stackFileType = 'jpg';
for i=stackSize:-1:1
  file = sprintf("%s%i.%s", stackBaseName, i-1, stackFileType);
  imgStack(:,:,:,i) = double(imread(file));
end
exposureTimes = [1/400, 1/250, 1/100, 1/40, 1/25, 1/8, 1/3];

%% Calculate curve
downSfactor = 25;
Z = reshape(imgStack(1:downSfactor:end,1:downSfactor:end,:),[],size(imgStack,3),stackSize);
B = log(exposureTimes);
l = 100;
Zmin = 0; Zmax = 255;
w = weightFunctions('tent',0,255,exposureTimes);

[g_r,~] = gsolve(squeeze(Z(:,1,:)),B,l,w); disp("Done red curve");
[g_g,~] = gsolve(squeeze(Z(:,2,:)),B,l,w); disp("Done green curve");
[g_b,~] = gsolve(squeeze(Z(:,3,:)),B,l,w); disp("Done blue curve");

% %% Remove outliers and correct with interpolation
[~,upper] = max([g_r,g_g,g_b]);
[~,lower] = min([g_r,g_g,g_b]);
g_r = g_r(lower(1):upper(1));
g_r = interp1((lower(1):upper(1)),g_r,(0:255),'linear','extrap');
g_g = g_g(lower(2):upper(2));
g_g = interp1((lower(2):upper(2)),g_g,(0:255),'linear','extrap');
g_b = g_b(lower(3):upper(3));
g_b = interp1((lower(3):upper(3)),g_b,(0:255),'linear','extrap');

% %% Plot results
figure; 
%clf; 
hold on;
plot(g_r, (0:255), 'r'); plot(g_g, (0:255), 'g'); plot(g_b, (0:255), 'b');
ylabel("Z_i_j"); xlabel("ln(E)");
legend(["Red curve","Green curve","Blue curve"], "Location","northwest");

%% Calibrate images
imgStackCalibrated(:,:,1,:) = exp(g_r(imgStack(:,:,1,:)+1));
imgStackCalibrated(:,:,2,:) = exp(g_g(imgStack(:,:,2,:)+1));
imgStackCalibrated(:,:,3,:) = exp(g_b(imgStack(:,:,3,:)+1));
imgStackCalibratedNorm = imgStackCalibrated/255.0;
radianceMap = mergeLDRStack(imgStackCalibratedNorm, exposureTimes, weightFunctions('tent',0.0,0.99));

%% Display image
figure; tiledlayout(3,1);
nexttile; imagesc(radianceMap(:,:,1)); colorbar; title("Red exposure");
nexttile; imagesc(radianceMap(:,:,2)); colorbar; title("Green exposure");
nexttile; imagesc(radianceMap(:,:,3)); colorbar; title("Blue exposure");

%%
figure;
tonedImage = toneMapping(radianceMap, 0.8);
imshow(tonedImage);