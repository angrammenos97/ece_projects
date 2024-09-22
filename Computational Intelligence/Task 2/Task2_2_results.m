%% Computational Intelligence Task 2.2
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
close all
clear
clc

%% Load results
load('Task2_2_output.mat');

%% Grid Search plots
figure('WindowState','maximized'); t = tiledlayout(1,2); title(t, 'Grid Search Results')
%Plot number of features
nexttile; plot(numberOfFeatures', gsRMSE); grid on;
xlabel('# of features'); ylabel('RMSE error'); legend("influence range " + radii);
%Plot influence range
nexttile; plot(radii', gsRMSE); grid on;
xlabel('Influence range'); ylabel('RMSE error'); legend("# of features " + radii);

%% Trained model results
figure('WindowState','maximized'); t = tiledlayout(2,1); title(t, 'SC Model Results')
%Plot predicted and actual output
nexttile; plot([scY Dval(:,end)]); grid on;
xlabel('Input Id'); ylabel('Output'); legend("Predicted Output", "Actual Output");
%Plot influence range
nexttile; plot([scTrnError scValError]); grid on;
xlabel('# of Iterations'); ylabel('RMSE error'); legend('Training Error','Validation Error');

%% Comparison initial and final member function
figure('WindowState','maximized'); t = tiledlayout(5,2); title(t, 'Member Functions');
%First row
nexttile; plotmf(scFis, 'input' , 1); grid on; title('Initial Model');
nexttile; plotmf(scValFis, 'input' , 1); grid on; title('Trained Model');
%Second row
nexttile; plotmf(scFis, 'input' , 2); grid on;
nexttile; plotmf(scValFis, 'input' , 2); grid on;
%Third row
nexttile; plotmf(scFis, 'input' , 3); grid on;
nexttile; plotmf(scValFis, 'input' , 3); grid on;
%Forth row
nexttile; plotmf(scFis, 'input' , 4); grid on;
nexttile; plotmf(scValFis, 'input' , 4); grid on;
%Fifth row
nexttile; plotmf(scFis, 'input' , 5); grid on;
nexttile; plotmf(scValFis, 'input' , 5); grid on;

%% Display Validation Results
varNames = {'RMSE', 'R2', 'NMSE', 'NDEI'};
resTable = array2table(modelPerfomance, 'VariableNames', varNames);
disp(resTable);

%time 1321.170652