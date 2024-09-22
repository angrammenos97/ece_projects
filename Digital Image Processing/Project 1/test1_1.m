clear
img = imread('TestIm1.png');
%% Rotation on TestIm1.png by theta=35degrees
angle = 35;
rotImg35 = myImgRotation(img, angle);
imwrite(rotImg35,"TestIm1_35o.png");
%% Rotation on TestIm1.png by theta=222degrees
angle = 222;
rotImg222 = myImgRotation(img, angle);
imwrite(rotImg222,"TestIm1_222o.png");