%% Computational Intelligence Task 4.3
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
close all
clear
clc

%% Load results
load('Task4_3_output.mat')

%% Display RMSE and R2
data = [rmse r2];
varNames = {'RMSE' 'R2'};
resTable = array2table(data, 'VariableNames', varNames);
disp(resTable);

%% Display learning curves
plot([loss' val_loss']); grid on; title('Loss');
xlabel('# of Epochs'); ylabel('Error'); legend('Training loss','Validation loss');
