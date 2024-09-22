clear
load("dip_hw_2.mat");
rng(1);

d2a_affi = Image2Graph(d2a);
d2b_affi = Image2Graph(d2b);

labels_d2a_k3 = myGraphSpectralClustering(d2a_affi, 3);
labels_d2a_k4 = myGraphSpectralClustering(d2a_affi, 4);
labels_d2b_k3 = myGraphSpectralClustering(d2b_affi, 3);
labels_d2b_k4 = myGraphSpectralClustering(d2b_affi, 4);

close all
figure; tiledlayout(1,3);
nexttile; imshow(segmentsColor(d2a, reshape(labels_d2a_k3,size(d2b,[1,2])))); title("K=3");
nexttile; imshow(d2a); title("Original");
nexttile; imshow(segmentsColor(d2a, reshape(labels_d2a_k4,size(d2b,[1,2])))); title("K=4");

figure; tiledlayout(1,3);
nexttile; imshow(segmentsColor(d2b, reshape(labels_d2b_k3,size(d2b,[1,2])))); title("K=3");
nexttile; imshow(d2b); title("Original");
nexttile; imshow(segmentsColor(d2b, reshape(labels_d2b_k4,size(d2b,[1,2])))); title("K=4");