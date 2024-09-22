%% Computational Intelligence Task 4.2
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
close all
clear
clc

%% Load results
load('Task4_2_10.mat')
load('Task4_2_50.mat')
load('Task4_2_90.mat')

%% Display RMSE and R2
data = [rmse_10 r2_10;...
        rmse_50 r2_50;...
        rmse_90 r2_90];
rowNames = {'10%', '50%', '90%'};
varNames = {'RMSE' 'R2'};
resTable = array2table(data, 'VariableNames', varNames, 'RowNames', rowNames);
disp(resTable);

%% Display learning curves
figure('WindowState','maximized'); t = tiledlayout(3,1); title(t, 'Learing Curves');
nexttile; plot([loss_10' val_loss_10']); grid on; title('Loss for 10%');
xlabel('# of Epochs'); ylabel('Error'); legend('Training loss','Validation loss');
nexttile; plot([loss_50' val_loss_50']); grid on; title('Loss for 50%');
xlabel('# of Epochs'); ylabel('Error'); legend('Training loss','Validation loss');
nexttile; plot([loss_90' val_loss_90']); grid on; title('Loss for 90%');
xlabel('# of Epochs'); ylabel('Error'); legend('Training loss','Validation loss');
