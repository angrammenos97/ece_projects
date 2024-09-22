clear;
clc;

%% Definitions
%Input
tstep = 0.0001;     %time step
tfinal = 10;        %time finale
max_ampl = 1;   %max input amplitude
max_angv = 10;  %max input angular velocity
num_sin = 10;   %number of sin/cos in input
%Method parameters
n = 9;  %output grade
m = 8;  %input grade
p = -1; %pole of the filter

%% Create training data
[u, tspan] = createInput([tstep, tfinal], max_ampl, max_angv, num_sin);   %sample input
y_N = out(tspan, u);   %sample output

%% Train model
theta = offlineLeastSquare(tspan, y_N, u, n, m, p);

%% Validate model
[u, ~] = createInput(tspan, max_ampl, max_angv, num_sin);   %sample input
y_N = out(tspan, u);   %sample output
y_M = simulateModel(theta, u, tspan, n);

%% Display results
plot(tspan, y_N, '-', tspan, y_M, ':', tspan, y_N - y_M, 'LineWidth', 3);
legend('y_N', 'y_M', 'error');
fprintf('(n,m,p)=(%d,%d,%g) absolute error summation= %g\n',...
                      n,m,p, mean(abs(y_M - y_N)));