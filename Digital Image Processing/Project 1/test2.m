clear
img1 = imread('TestIm1.png');
img2 = imread('TestIm2.png');

tic; imgS = myStitch(img1, img2); toc;
imwrite(imgS,"TestIm_merged.png");

imshow(imgS);