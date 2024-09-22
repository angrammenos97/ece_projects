clc;
clear;

m = 15.0;   %kg
b = 0.2;    %kg/sec
k = 2.0;    %kp/sec^2
%u = 5*sin(2*t) + 10.5;  %N
step = 0.1;     %sec
time_f = 10.0;  %sec
tspan = 0.0:step:time_f;
states0 = [0.0, 0.0];    %initial state

%Sample output y
F = @(t,states)dynamics(t, states, b, k, m);
options = odeset('RelTol',10^-10,'AbsTol',10^-11);
[t, states] = ode15s(F, tspan, states0, options);
y_N = states(:,1);

%Linear Model
L = tf([1 3 2],1);  %from filter L(s) = s^2 + 3s + 2
theta_l = [(b/m - 3);(k/m - 2);1];
zeta = [ tf([-1 0],1)/L ; -1/L ; 1/L];

%Least Square 
[u,t] = gensig('sin', pi, time_f, step);
u = 5.0*u + 10.5;   %N
zeta_N = [ lsim(zeta(1), y_N, t), lsim(zeta(2), y_N, t), lsim(zeta(3), u, t)];
theta_l_zero = linsolve(zeta_N'*zeta_N , zeta_N'*y_N);
