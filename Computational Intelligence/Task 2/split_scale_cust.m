%% Computational Intelligence Task 1
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    28/09/2021

%% Split_scale Function
%Scales data to follow Standard Normal Distribution
%and spilts it to traing, check and test data by 60%, 20% and 20%

%Inputs:
%data -> Matrix where columns are inputs/output

%Outputs:
%trnData -> Matrix with training data
%chkData -> Matrix with check data
%tstData -> Matrix with test data

function [trnData, chkData, tstData] = split_scale_cust(data)
    % Move and scale for mean = 0 and std = 1
    scaledData = (data - mean(data)) ./ std(data);
    % Training data is the 60% of the total data
    trnDataPercentage = floor(size(data, 1)*0.6);
    trnData = scaledData(1:trnDataPercentage, :);
    % Check data is the 20% of the total data
    chkDataPercentage = floor(size(data, 1)*0.2);
    chkData = scaledData(trnDataPercentage+1:trnDataPercentage+chkDataPercentage, :);
    % Test data is the 20% of the total data
    tstData = scaledData(trnDataPercentage+chkDataPercentage+1:end, :);
end