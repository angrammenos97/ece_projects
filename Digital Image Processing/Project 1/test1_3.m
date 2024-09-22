clear;
img = imread('TestIm1.png');
%img = imread('TestIm2.png');
img = myImgRotation(img, 0);
%img = checkerboard(50);
%% Find corners
grayImg = rgb2gray(img);
%grayImg = img;
normGrayImg = double(grayImg)/255.0;
tic; corners = myDetectHarrisFeatures(normGrayImg); toc;
points = detectHarrisFeatures(grayImg);
%% Draw red 5x5 boxes on corners
imgSize = size(img);
boxedImg = drawSquares(img, corners, 5);
%% Display results
imshow(boxedImg); hold on;
plot(points.selectStrongest(50));