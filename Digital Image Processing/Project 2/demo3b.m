clear
load("dip_hw_2.mat");
rng(1);

T1 = 5;
T2 = 0.6;

d2a_affi = Image2Graph(d2a);
labels_d2a = ones(size(d2a_affi,1),1);
[labels_d2a,clustersNum_d2a]=nCuts(d2a_affi, labels_d2a, T1, T2);

d2b_affi = Image2Graph(d2b);
labels_d2b = ones(size(d2b_affi,1),1);
[labels_d2b,clustersNum_d2b]=nCuts(d2b_affi, labels_d2b, T1, T2);

close all;
figure; tiledlayout(1,2);
nexttile; imshow(d2a); title("Original");
nexttile; imshow(segmentsColor(d2a, reshape(labels_d2a,size(d2a,[1,2])))); 
title("k="+clustersNum_d2a);

figure; tiledlayout(1,2);
nexttile; imshow(d2b); title("Original");
nexttile; imshow(segmentsColor(d2b, reshape(labels_d2b,size(d2b,[1,2])))); 
title("k="+clustersNum_d2b);
