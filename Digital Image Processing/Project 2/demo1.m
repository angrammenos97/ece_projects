clear
load("dip_hw_2.mat");

rng(1);
lables_k2 = myGraphSpectralClustering(d1a, 2);
lables_k3 = myGraphSpectralClustering(d1a, 3);
lables_k4 = myGraphSpectralClustering(d1a, 4);