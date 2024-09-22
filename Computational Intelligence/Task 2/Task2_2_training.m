%% Computational Intelligence Task 2.2
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
clear
clc

%% Load data
load('Task2_2_output.mat');

%% Choose parameters
%Decided from results of the gsRMSE array of Task2_2_grid_output.mat file
prefNumOfFeat = 50;
prefRadii = 0.4;

%% Choose features of the data
Dtrn = [Dtrn(:,1:prefNumOfFeat) Dtrn(:,end)];
Dchk = [Dchk(:,1:prefNumOfFeat) Dchk(:,end)];
Dval = [Dval(:,1:prefNumOfFeat) Dval(:,end)];

%% Subtractive Clustering TSK model FIS options
scFisOpt = genfisOptions('SubtractiveClustering');
scFisOpt.ClusterInfluenceRange = prefRadii;
scFisOpt.Verbose = 0;

%% ANFIS options
anfisOpt = anfisOptions;
anfisOpt.EpochNumber = 100;
anfisOpt.DisplayANFISInformation = 0;
anfisOpt.DisplayErrorValues = 0;
anfisOpt.DisplayStepSize = 1;
anfisOpt.DisplayFinalResults = 0;
anfisOpt.ValidationData = Dchk;
anfisOpt.OptimizationMethod = 1;

%% Evaluation functions
RMSE = @(ypred, y) sqrt(sum((y-ypred).^2) / size(ypred, 1));
R2   = @(ypred, y) 1 - (sum((y-ypred).^2) / sum((y-mean(y)).^2));
NMSE = @(ypred, y) sum((y-ypred).^2) / sum((y-mean(y)).^2);
NDEI = @(ypred, y) sqrt(NMSE(ypred, y));

%% Train the two models
%SC TSK model
disp("Training SC model, please wait...");
scFis = genfis(Dtrn(:,1:end-1), Dtrn(:,end), scFisOpt);
scAnfisOpt = anfisOpt;
scAnfisOpt.InitialFIS = scFis;
tic;
[~, scTrnError, ~, scValFis, scValError] = anfis(Dtrn, scAnfisOpt);
toc;
scY = evalfis(scValFis, Dval(:, 1:end-1));

%% Evaluate the models
modelPerfomance = zeros(1,4);
%SC TSK model evaluation
modelPerfomance(1) = RMSE(scY, Dval(:,end));
modelPerfomance(2) = R2(scY, Dval(:,end));
modelPerfomance(3) = NMSE(scY, Dval(:,end));
modelPerfomance(4) = NDEI(scY, Dval(:,end));

%% Save results
save('Task2_2_output.mat', '-append', 'prefNumOfFeat', 'prefRadii', ...
    'scFis', 'scValFis', 'scTrnError', 'scValError', ...
    'scY', 'modelPerfomance');
