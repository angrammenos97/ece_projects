clear;
clc;

%% Definitions
tstep = 0.0001; %time step
tfinal = 10;    %time finale
max_ampl = 1;   %max input amplitude
max_angv = 10;  %max input angular velocity
num_sin = 10;   %number of sin/cos in input
a = 2;          %homogeneity factor

%% Initialize
% Sample inputs
[u1, tspan] = createInput([tstep, tfinal], max_ampl, max_angv, num_sin);       
[u2, ~] = createInput(tspan, max_ampl, max_angv, num_sin);
% Sample outputs
y1 = out(tspan, u1);
y2 = out(tspan, u2);
y1y2 = out(tspan, u1+ u2);
ay1 = out(tspan, a*u1);

%% Check additivity
add_error = sum(abs(y1y2 - (y1+y2)));

%% Check homogeneity
homo_error = sum(abs(ay1 - (a*y1)));

%% Display errors
fprintf('Additivity error= %g\n', add_error);
fprintf('Homogeneity error = %g\n', homo_error);