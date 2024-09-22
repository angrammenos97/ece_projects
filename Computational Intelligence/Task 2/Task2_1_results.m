%% Computational Intelligence Task 2.1
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
close all
clear
clc

%% Load results
load('Task2_1_output.mat');

%% Plot Member Functions
figure('WindowState','maximized'); t = tiledlayout(5,4); title(t, 'Member Functions');
%First row
nexttile; plotmf(valFis1, 'input' , 1); grid on; title('TSK model 1');
nexttile; plotmf(valFis2, 'input' , 1); grid on; title('TSK model 2');
nexttile; plotmf(valFis3, 'input' , 1); grid on; title('TSK model 3');
nexttile; plotmf(valFis4, 'input' , 1); grid on; title('TSK model 4');
%Second row
nexttile; plotmf(valFis1, 'input' , 2); grid on; nexttile; plotmf(valFis2, 'input' , 2); grid on;
nexttile; plotmf(valFis3, 'input' , 2); grid on; nexttile; plotmf(valFis4, 'input' , 2); grid on;
%Third row
nexttile; plotmf(valFis1, 'input' , 3); grid on; nexttile; plotmf(valFis2, 'input' , 3); grid on;
nexttile; plotmf(valFis3, 'input' , 3); grid on; nexttile; plotmf(valFis4, 'input' , 3); grid on;
%Forth row
nexttile; plotmf(valFis1, 'input' , 4); grid on; nexttile; plotmf(valFis2, 'input' , 4); grid on;
nexttile; plotmf(valFis3, 'input' , 4); grid on; nexttile; plotmf(valFis4, 'input' , 4); grid on;
%Fifth row
nexttile; plotmf(valFis1, 'input' , 5); grid on; nexttile; plotmf(valFis2, 'input' , 5); grid on;
nexttile; plotmf(valFis3, 'input' , 5); grid on; nexttile; plotmf(valFis4, 'input' , 5); grid on;

%% Plot Training and Validation Error
figure('WindowState','maximized'); t = tiledlayout(3,2); title(t, 'Traing and Validation Errors');
%TSK model 1
nexttile; plot([trnError1 valError1]); grid on; title('TSK model 1');
xlabel('# of Iterations'); ylabel('Error'); legend('Training Error','Validation Error');
%TSK model 2
nexttile; plot([trnError2 valError2]); grid on; title('TSK model 2');
xlabel('# of Iterations'); ylabel('Error'); legend('Training Error','Validation Error');
%TSK model 3
nexttile; plot([trnError3 valError3]); grid on; title('TSK model 3');
xlabel('# of Iterations'); ylabel('Error'); legend('Training Error','Validation Error');
%TSK model 1
nexttile; plot([trnError4 valError4]); grid on; title('TSK model 4');
xlabel('# of Iterations'); ylabel('Error'); legend('Training Error','Validation Error');
%All TSK models
nexttile(5, [1 2]); plot([valError1 valError2 valError3 valError4]); grid on; title('All models');
xlabel('# of Iterations'); ylabel('Error'); 
legend('Validation Error 1', 'Validation Error 2', 'Validation Error 3', 'Validation Error 4');

%% Display Validation Results
res = [ ...
    RMSE1, NMSE1, NDEI1, R21, trnTime1; ...
    RMSE2, NMSE2, NDEI2, R22, trnTime2; ...
    RMSE3, NMSE3, NDEI3, R23, trnTime3; ...
    RMSE2, NMSE4, NDEI4, R24, trnTime4 ]';
varNames = {'TSK model 1', 'TSK model 2', 'TSK model 3', 'TSK model 4'};
rowNames = {'RMSE', 'NMSE', 'NDEI', 'R2', 'El. time'};
resTable = array2table(res,'VariableNames',varNames,'RowNames',rowNames);
disp(resTable);