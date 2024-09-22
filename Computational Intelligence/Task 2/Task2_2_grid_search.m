%% Computational Intelligence Task 2.2
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
clear
clc

%% Load data - Split data
data = table2array(readtable('train.csv'));      %Load data
featIdx = fscmrmr(data(:,1:end-1), data(:,end)); %Use mRMR to rank features
data = [data(:,featIdx), data(:,end)];           %Reorder features
[Dtrn, Dchk, Dval] = split_scale(data, 1);       %Shuffle, scale and split dataset

%% Define Free Parameters
numberOfFeatures = [10 20 40 50 60 80];
radii = [0.8 0.7 0.6 0.5 0.4];  %Range of influence of the cluster center

%% Define global FIS and learing options
%FIS Options
fisOpt = genfisOptions('SubtractiveClustering');
fisOpt.Verbose = 0;
%ANFIS Options
anfisOpt = anfisOptions;
anfisOpt.EpochNumber = 10;
anfisOpt.DisplayANFISInformation = 0;
anfisOpt.DisplayErrorValues = 0;
anfisOpt.DisplayStepSize = 0;
anfisOpt.DisplayFinalResults = 0;
%K-Fold Cross Validation Options
kFoldValue = 5;

%% Grid Search
gsRMSE = zeros(length(numberOfFeatures), length(radii));
numOfRules = zeros(length(numberOfFeatures), length(radii));
tic
parfor i = 0:length(numberOfFeatures)*length(radii)-1
    %Extract point's parameters
    numOfFeatIdx = floor(i/length(radii))+1;
    radiiIdx = mod(i,length(radii))+1;
    %Set Cluster Influence Range
    iFisOpt = fisOpt;
    iFisOpt.ClusterInfluenceRange = radii(radiiIdx);
    %Run K-Fold Cross Validation on ANFIS
    dChkSz = round(length(Dtrn)/kFoldValue); %Size of validation dataset
    valErrors = zeros(1,kFoldValue);
    inumOfRules = zeros(1,kFoldValue);
    fis = [];
    for k = 0:kFoldValue-1
        %Split training and validation data for K-Fold CV
        [cvDtrn,cvDchk] = deal([Dtrn(:,1:numberOfFeatures(numOfFeatIdx)) Dtrn(:,end)]);
        cvDchk((k+1)*dChkSz:end,:) = [];                     %Remove training data
        cvDtrn((k*dChkSz)+1:min((k+1)*dChkSz-1,end),:) = []; %Remove validation data
        %Generate FIS and set options for ANFIS
        fis = genfis(cvDtrn(:,1:end-1), cvDtrn(:,end),iFisOpt);
        inumOfRules(k+1) = length(fis.Rules);
        iAnfisOpt = anfisOpt;
        iAnfisOpt.InitialFIS = fis;
        iAnfisOpt.ValidationData = cvDchk;   
        %Train model
        [~,~,~,~,valError] = anfis(cvDtrn, iAnfisOpt);
        valErrors(k+1) = min(valError); %Select the minimum validation RMSE
    end
    %Save the mean of RMSE errors of this point and number of rules
    gsRMSE(i+1) = mean(valErrors);
    numOfRules(i+1) = mean(inumOfRules);
end
toc

%% Save results
save('Task2_2_output.mat', 'Dtrn', 'Dchk', 'Dval', ...
    'numberOfFeatures', 'radii', 'gsRMSE', 'numOfRules');