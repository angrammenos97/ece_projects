clear;
clc;

% Specifications
a = 2;
b = 1;
u = @(t) 5*sin(3*t);
heta0 = 0.60;   %noise amplitude
f0 = 20;        %noise freq
time_s = 0.0;   %start time
time_f = 100.0; %end time

% Initialize
X0 = 0;
a0 = 5;     %random
b0 = 10;    %random
vars0 = [X0, X0, a0, b0]; % 1:X 2:X_hat 3:a_hat 4:b_hat

% Lyapunov method
tspan = [time_s time_f];
options = odeset('RelTol',10^-10,'AbsTol',10^-11);
%1. Parallel conf
Sys_p = @(t_p, vars_p)derivative_sys_p(t_p, vars_p, a, b, u, heta0, f0);
[t_p, vars_p] = ode45(Sys_p, tspan, vars0, options);
%2. Series-Parallel
theta_m = 2;    %for series-parallel conf
Sys_m = @(t_m, vars_m)derivative_sys_m(t_m, vars_m, a, b, u, theta_m, heta0, f0);
[t_m, vars_m] = ode45(Sys_m, tspan, vars0, options);

% Visualize
figure(1);
plot(t_p, vars_p(:,1), t_p, vars_p(:,2));
legend("x real", "x-hat parallel");
figure(2);
plot(t_p, vars_p(:,3), t_p, vars_p(:,4));
legend("a-hat parallel", "b-hat parallel");
figure(3);
plot(t_m, vars_m(:,1), t_m, vars_m(:,2));
legend("x real", "x-hat mixed");
figure(4);
plot(t_m, vars_m(:,3), t_m, vars_m(:,4));
legend("a-hat mixed", "b-hat mixed");

function dvars_p = derivative_sys_p(t_p, vars_p, a, b, u, heta0, f0)
    dvars_p(1) = -a*vars_p(1) + b*u(t_p);                   %X
    measurement = vars_p(1) + heta0*sin(2*f0*pi*t_p);
    %Paraller conf
    dvars_p(2) = -vars_p(3)*vars_p(2) + vars_p(4)*u(t_p);   %X_hat_p
    dvars_p(3) = vars_p(2)^2 - measurement*vars_p(2);       %a_hat_p
    dvars_p(4) = (measurement - vars_p(2))*u(t_p);          %b_hat_p
    
    dvars_p = dvars_p';
end

function dvars_m = derivative_sys_m(t_m, vars_m, a, b, u, theta_m, heta0, f0)
    dvars_m(1) = -a*vars_m(1) + b*u(t_m);                   %X
    measurement = vars_m(1) + heta0*sin(2*f0*pi*t_m);
    %Series-Paraller conf
    dvars_m(2) = -vars_m(3)*measurement + vars_m(4)*u(t_m) ...
                + theta_m*(measurement - vars_m(2));        %X_hat_m
    dvars_m(3) = -measurement^2 + measurement*vars_m(2);    %a_hat_m
    dvars_m(4) = (measurement - vars_m(2))*u(t_m);          %b_hat_m
    
    dvars_m = dvars_m';
end