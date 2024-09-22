clear;
clc;

%% Definitions
%For training
tstept = 0.0001;     %time step
tfinalt = 10;        %time finale
max_amplt = 1;   %max input amplitude
max_angvt = 100;  %max input angular velocity
num_sint = 1;   %number of sin/cos in input
%For validation
tstepv = 0.0001;     %time step
tfinalv = 10;        %time finale
max_amplv = 1;   %max input amplitude
max_angvv = 100;  %max input angular velocity
num_sinv = 100;   %number of sin/cos in input

%% Each method's parameters
%Offline
noff = 3;  %output grade
moff = 2;  %input grade
poff = -1; %pole of the filter
%Online
non = 3;  %output grade
mon = 2;  %input grade
pon = -1; %pole of the filter
b = 5;    %beta of the algorithm
Q0 = eye(non+mon+1);    %Q_zero of the algorithm

%% Create training data
[u, tspan] = createInput([tstept, tfinalt], max_amplt, max_angvt, num_sint);   %sample input
y_N = out(tspan, u);   %sample output

%% Train models
thetaoff = offlineLeastSquare(tspan, y_N, u, noff, moff, poff);
thetaon = onlineLeastSquare(tspan, y_N, u, non, mon, pon, b, Q0);

%% Validate models
[u, tspan] = createInput([tstepv, tfinalv], max_amplv, max_angvv, num_sinv);   %sample input
y_N = out(tspan, u);   %sample output
y_Moff = simulateModel(thetaoff, u, tspan, noff);
y_Mon = simulateModel(thetaon(:,end), u, tspan, non);

%% Display results
plot(tspan, y_N, '-', tspan, y_Moff, '--', tspan, y_Mon, ':', 'LineWidth', 3);
legend('y_N', 'y_Moff', 'y_Mon');
fprintf('Offline: absolute error summation= %g\n',...
                      sum(abs(y_Moff - y_N)));
fprintf('Online : absolute error summation= %g\n',...
                      sum(abs(y_Mon - y_N)));