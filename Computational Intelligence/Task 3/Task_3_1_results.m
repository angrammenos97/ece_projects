%% Computational Intelligence Task 3.1
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
close all
clear
clc

%% Load results
load('Task3_1_code/Task3_1_default_b1.mat')
load('Task3_1_code/Task3_1_default_b256.mat')
load('Task3_1_code/Task3_1_default_b70000.mat')

load('Task3_1_code/Task3_1_RMS_rho01.mat')
load('Task3_1_code/Task3_1_RMS_rho99.mat')

load('Task3_1_code/Task3_1_SGD_mean10.mat')

load('Task3_1_code/Task3_1_SGD_l2_1.mat')
load('Task3_1_code/Task3_1_SGD_l2_01.mat')
load('Task3_1_code/Task3_1_SGD_l2_001.mat')

load('Task3_1_code/Task3_1_drop_l1.mat')

%% Display learning curves
figure('WindowState','maximized'); t = tiledlayout(3,2); title(t, 'Default net');
%Batch size = 1
nexttile; plot([default_b1_accuracy' default_b1_val_accuracy']); grid on; title('Accuracy, Batch size = 1');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([default_b1_loss' default_b1_val_loss']); grid on; title('Loss, Batch size = 1');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
%Batch size = 256
nexttile; plot([default_b256_accuracy' default_b256_val_accuracy']); grid on; title('Accuracy, Batch size = 256');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([default_b256_loss' default_b256_val_loss']); grid on; title('Loss, Batch size = 256');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
%Batch size = all
nexttile; plot([default_b70000_accuracy' default_b70000_val_accuracy']); grid on; title('Accuracy, Batch size = all');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([default_b70000_loss' default_b70000_val_loss']); grid on; title('Loss, Batch size = all');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');

figure('WindowState','maximized'); t = tiledlayout(3,2); title(t, 'RMSProp optimizer with lr = 0.001');
%roh = 0.01
nexttile; plot([RMS_rho01_accuracy' RMS_rho01_val_accuracy']); grid on; title('Accuracy, ρ = 0.01');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([RMS_rho01_loss' RMS_rho01_val_loss']); grid on; title('Loss, ρ = 0.01');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
%roh = 0.99
nexttile; plot([RMS_rho99_accuracy' RMS_rho99_val_accuracy']); grid on; title('Accuracy, ρ = 0.99');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([RMS_rho99_loss' RMS_rho99_val_loss']); grid on; title('Loss, ρ = 0.99');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');

figure('WindowState','maximized'); t = tiledlayout(4,2); title(t, 'SGD optimizer with lr = 0.01 and initial weights with mean = 10');
%No L2-norm
nexttile; plot([SGD_mean10_accuracy' SGD_mean10_val_accuracy']); grid on; title('Accuracy, No L2-norm');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([SGD_mean10_loss' SGD_mean10_val_loss']); grid on; title('Loss, No L2-norm');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
%a=0.1
nexttile; plot([SGD_l2_1_accuracy' SGD_l2_1_val_accuracy']); grid on; title('Accuracy, α = 0.1');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([SGD_l2_1_loss' SGD_l2_1_val_loss']); grid on; title('Loss, α = 0.1');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
%a=0.01
nexttile; plot([SGD_l2_01_accuracy' SGD_l2_01_val_accuracy']); grid on; title('Accuracy, α = 0.01');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([SGD_l2_01_loss' SGD_l2_01_val_loss']); grid on; title('Loss, α = 0.01');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
%a=0.001
nexttile; plot([SGD_l2_001_accuracy' SGD_l2_001_val_accuracy']); grid on; title('Accuracy, α = 0.001');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([SGD_l2_001_loss' SGD_l2_001_val_loss']); grid on; title('Loss, α = 0.001');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');

figure('WindowState','maximized'); t = tiledlayout(3,2); title(t, 'Default net with dropout and L1-norm');
nexttile; plot([Drop_l1_accuracy' Drop_l1_val_accuracy']); grid on; title('Accuracy, α = 0.01 , dropout probability = 0.3');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy', 'Location', 'best');
nexttile; plot([Drop_l1_loss' Drop_l1_val_loss']); grid on; title('Loss, α = 0.01 , dropout probability = 0.3');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss', 'Location', 'best');
