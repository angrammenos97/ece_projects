%% Computational Intelligence Task 3.2
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022
close all
clear
clc

%% Load results
load('Task3_2_output.mat')

%% Calculate Precision, Recall and F-Measure
cm = double(confusion_matrix);
accuracy_m = sum(diag(cm))/sum(cm, 'all');
precision_m = mean(diag(cm)' ./ sum(cm, 2)');
recall_m = mean(diag(cm)' ./ sum(cm, 1));
fmeasure_m = 2*precision_m*recall_m/(precision_m+recall_m);

%% Display Confusion Matrix
figure;
confusionchart(confusion_matrix, 0:1:9)
title('Confusion Matrix');

%% Display Accuracy, Precision, Recall and F-Measure
data = [accuracy_m precision_m recall_m fmeasure_m];
varNames = {'Accuracy' 'Precision' 'Recall' 'F-Measure'};
resTable = array2table(data, 'VariableNames', varNames);
disp(resTable);

%% Display learning curves
figure('WindowState','maximized'); t = tiledlayout(2,1); title(t, 'Learing Curves');
nexttile; plot([loss' val_loss']); grid on; title('Loss');
xlabel('# of Iterations'); ylabel('Error'); legend('Training loss','Validation loss');
nexttile; plot([accuracy' val_accuracy']); grid on; title('Accuracy');
xlabel('# of Iterations'); ylabel('Error'); legend('Training accuracy','Validation accuracy');