%% Computational Intelligence Task 2.1
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
clear
clc

%% Load data - Split data
data = load('airfoil_self_noise.dat');
[Dtrn, Dchk, Dval] = split_scale(data, 1);

%% Global Common Settings
%FIS Options
fisClusteringType = 'GridPartition';
fisInputMembershipFunctionType = 'gbellmf';
%ANFIS Options
anfisEpochNumber = 100;
anfisDisplayANFISInformation = 0;
anfisDisplayErrorValues = 0;
anfisDisplayStepSize = 0;
anfisDisplayFinalResults = 0;
anfisValidationData = Dchk;
anfisOptimizationMethod = 1;
%Save Options
save('Task2_1_output.mat', 'anfisEpochNumber');

%% Evaluation functions
RMSE = @(ypred, y) sqrt(sum((y-ypred).^2) / size(ypred, 1));
R2   = @(ypred, y) 1 - (sum((y-ypred).^2) / sum((y-mean(y)).^2));
NMSE = @(ypred, y) sum((y-ypred).^2) / sum((y-mean(y)).^2);
NDEI = @(ypred, y) sqrt(NMSE(ypred, y));

%% TSK_model_1
%FIS Options
fisOpt1 = genfisOptions(fisClusteringType);
fisOpt1.NumMembershipFunctions = 2;
fisOpt1.InputMembershipFunctionType = fisInputMembershipFunctionType;
fisOpt1.OutputMembershipFunctionType = 'constant';
%Generate FIS
fis1 = genfis(Dtrn(:,1:5), Dtrn(:,6), fisOpt1);
%ANFIS Options
anfisOpt1 = anfisOptions;
anfisOpt1.InitialFIS = fis1;
anfisOpt1.EpochNumber = anfisEpochNumber;
anfisOpt1.DisplayANFISInformation = anfisDisplayANFISInformation;
anfisOpt1.DisplayErrorValues = anfisDisplayErrorValues;
anfisOpt1.DisplayStepSize = anfisDisplayStepSize;
anfisOpt1.DisplayFinalResults = anfisDisplayFinalResults;
anfisOpt1.ValidationData = anfisValidationData;
anfisOpt1.OptimizationMethod = anfisOptimizationMethod;
%Train ANFIS
disp("Training TSK_model_1, please wait...");
tic;
[trnFis1, trnError1, ~, valFis1, valError1] = anfis(Dtrn, anfisOpt1);
trnTime1 = toc;
disp("Elapsed time " + trnTime1);
%Evaluate
warning('off');
Y = evalfis(valFis1, Dval(:, 1:end-1));
warning('on');
RMSE1 = RMSE(Y, Dval(:,end));
R21   = R2(Y, Dval(:,end));
NMSE1 = NMSE(Y, Dval(:,end));
NDEI1 = NDEI(Y, Dval(:,end));
%Save Results
save('Task2_1_output.mat', '-append', 'valFis1', 'trnError1', 'valError1', 'trnTime1', 'RMSE1', 'R21', 'NMSE1', 'NDEI1');

%% TSK_model_2
%FIS Options
fisOpt2 = genfisOptions(fisClusteringType);
fisOpt2.NumMembershipFunctions = 3;
fisOpt2.InputMembershipFunctionType = fisInputMembershipFunctionType;
fisOpt2.OutputMembershipFunctionType = 'constant';
%Generate FIS
fis2 = genfis(Dtrn(:,1:5), Dtrn(:,6), fisOpt2);
%ANFIS Options
anfisOpt2 = anfisOptions;
anfisOpt2.InitialFIS = fis2;
anfisOpt2.EpochNumber = anfisEpochNumber;
anfisOpt2.DisplayANFISInformation = anfisDisplayANFISInformation;
anfisOpt2.DisplayErrorValues = anfisDisplayErrorValues;
anfisOpt2.DisplayStepSize = anfisDisplayStepSize;
anfisOpt2.DisplayFinalResults = anfisDisplayFinalResults;
anfisOpt2.ValidationData = anfisValidationData;
anfisOpt2.OptimizationMethod = anfisOptimizationMethod;
%Train ANFIS
disp("Training TSK_model_2, please wait...");
tic;
[trnFis2, trnError2, ~, valFis2, valError2] = anfis(Dtrn, anfisOpt2);
trnTime2 = toc;
disp("Elapsed time " + trnTime2);
%Evaluate
warning('off');
Y = evalfis(valFis2, Dval(:, 1:end-1));
warning('on');
RMSE2 = RMSE(Y, Dval(:,end));
R22   = R2(Y, Dval(:,end));
NMSE2 = NMSE(Y, Dval(:,end));
NDEI2 = NDEI(Y, Dval(:,end));
%Save Results
save('Task2_1_output.mat', '-append', 'valFis2', 'trnError2', 'valError2', 'trnTime2', 'RMSE2', 'R22', 'NMSE2', 'NDEI2');

%% TSK_model_3
%FIS Options
fisOpt3 = genfisOptions(fisClusteringType);
fisOpt3.NumMembershipFunctions = 2;
fisOpt3.InputMembershipFunctionType = fisInputMembershipFunctionType;
fisOpt3.OutputMembershipFunctionType = 'linear';
%Generate FIS
fis3 = genfis(Dtrn(:,1:5), Dtrn(:,6), fisOpt3);
%ANFIS Options
anfisOpt3 = anfisOptions;
anfisOpt3.InitialFIS = fis3;
anfisOpt3.EpochNumber = anfisEpochNumber;
anfisOpt3.DisplayANFISInformation = anfisDisplayANFISInformation;
anfisOpt3.DisplayErrorValues = anfisDisplayErrorValues;
anfisOpt3.DisplayStepSize = anfisDisplayStepSize;
anfisOpt3.DisplayFinalResults = anfisDisplayFinalResults;
anfisOpt3.ValidationData = anfisValidationData;
anfisOpt3.OptimizationMethod = anfisOptimizationMethod;
%Train ANFIS
disp("Training TSK_model_3, please wait...");
tic;
[trnFis3, trnError3, ~, valFis3, valError3] = anfis(Dtrn, anfisOpt3);
trnTime3 = toc;
disp("Elapsed time " + trnTime3);
%Evaluate
warning('off');
Y = evalfis(valFis3, Dval(:, 1:end-1));
warning('on');
RMSE3 = RMSE(Y, Dval(:,end));
R23   = R2(Y, Dval(:,end));
NMSE3 = NMSE(Y, Dval(:,end));
NDEI3 = NDEI(Y, Dval(:,end));
%Save Results
save('Task2_1_output.mat', '-append', 'valFis3', 'trnError3', 'valError3', 'trnTime3', 'RMSE3', 'R23', 'NMSE3', 'NDEI3');

%% TSK_model_4
%FIS Options
fisOpt4 = genfisOptions(fisClusteringType);
fisOpt4.NumMembershipFunctions = 3;
fisOpt4.InputMembershipFunctionType = fisInputMembershipFunctionType;
fisOpt4.OutputMembershipFunctionType = 'linear';
%Generate FIS
fis4 = genfis(Dtrn(:,1:5), Dtrn(:,6), fisOpt4);
%ANFIS Options
anfisOpt4 = anfisOptions;
anfisOpt4.InitialFIS = fis4;
anfisOpt4.EpochNumber = anfisEpochNumber;
anfisOpt4.DisplayANFISInformation = anfisDisplayANFISInformation;
anfisOpt4.DisplayErrorValues = anfisDisplayErrorValues;
anfisOpt4.DisplayStepSize = anfisDisplayStepSize;
anfisOpt4.DisplayFinalResults = anfisDisplayStepSize;
anfisOpt4.ValidationData = anfisValidationData;
anfisOpt4.OptimizationMethod = anfisOptimizationMethod;
%Train ANFIS
disp("Training TSK_model_4, please wait...");
tic;
[trnFis4, trnError4, ~, valFis4, valError4] = anfis(Dtrn, anfisOpt4);
trnTime4 = toc;
disp("Elapsed time " + trnTime4);
%Evaluate
warning('off');
Y = evalfis(valFis4, Dval(:, 1:end-1));
warning('on');
RMSE4 = RMSE(Y, Dval(:,end));
R24   = R2(Y, Dval(:,end));
NMSE4 = NMSE(Y, Dval(:,end));
NDEI4 = NDEI(Y, Dval(:,end));
%Save Results
save('Task2_1_output.mat', '-append', 'valFis4', 'trnError4', 'valError4', 'trnTime4', 'RMSE4', 'R24', 'NMSE4', 'NDEI4');
