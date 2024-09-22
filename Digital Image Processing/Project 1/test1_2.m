clear;
img = imread('TestIm1.png');
rotImg35 = imread("TestIm1_35o.png");
rotImg222 = imread("TestIm1_222o.png");
%% Apply on original pixel 100,100
tic
d1_0 = myLocalDescriptor(rgb2gray(img), [100,100], 5, 20, 1, 8);
toc
d1_35 = myLocalDescriptor(rgb2gray(rotImg35), [736,139], 5, 20, 1, 8);
d1_222 = myLocalDescriptor(rgb2gray(rotImg222), [905,1600], 5, 20, 1, 8);
error1_35 = mean(abs(d1_0-d1_35));   error1_222 = mean(abs(d1_0-d1_222));
fprintf("Mean error for 35degrees is %f and for 222degrees %f\n", error1_35, error1_222);
%% Apply on original pixel 200,200 and 202,202
d2_200 = myLocalDescriptor(rgb2gray(img), [200,200], 5, 20, 1, 8);
d2_202 = myLocalDescriptor(rgb2gray(img), [202,202], 5, 20, 1, 8);
% fprintf("Difference between (200,200) and (202,202) is %f\n");
% disp((d2_200-d2_202)');
%% Apply on upgraded descriptor
%Apply on rotated images
tic
du1_0 = myLocalDescriptorUpgrade(rgb2gray(img), [100,100], 5, 20, 1, 8);
toc
du1_35 = myLocalDescriptorUpgrade(rgb2gray(rotImg35), [736,139], 5, 20, 1, 8);
du1_222 = myLocalDescriptorUpgrade(rgb2gray(rotImg222), [905,1600], 5, 20, 1, 8);
erroru1_35 = mean(abs(du1_0-du1_35));   erroru1_222 = mean(abs(du1_0-du1_222));
fprintf("Mean error for 35degrees is %f and for 222degrees %f\n", erroru1_35, erroru1_222);
%Apply on original image
du2_200 = myLocalDescriptorUpgrade(rgb2gray(img), [200,200], 5, 20, 1, 8);
du2_202 = myLocalDescriptorUpgrade(rgb2gray(img), [202,202], 5, 20, 1, 8);
% fprintf("Difference between (200,200) and (202,202) is");
% disp((du2_200-du2_202)');