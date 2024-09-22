clear;
clc;

% Specifications
a = 2;
b = 1;
u = @(t) 5*sin(3*t);
time_s = 0.0;   %start time
time_f = 10.0;  %end time

% Initialize
X0 = 0;
f1_0 = 0;
f2_0 = 0;
theta1_0 = 5;   %random
theta2_0 = 10;  %random
vars0 = [X0, f1_0, f2_0, theta1_0, theta2_0];

% Gradient method
tspan = [time_s time_f];
l_m = 5;    %filter
g = 100;     %convergence speed
Sys = @(t, vars)derivative_sys(t, vars, a, b, u, l_m, g);
options = odeset('RelTol',10^-10,'AbsTol',10^-11);
[t, vars] = ode45(Sys, tspan, vars0, options);

% Visualize
x_hat = diag([vars(:,4), vars(:,5)]* [vars(:,2), vars(:,3)]');
figure(1);
plot(t, x_hat, t, vars(:,1));
legend("x-hat", "x-real");
figure(2);
plot(t, l_m-vars(:,4), t, vars(:,5));
legend("a-hat", "b-hat");

function dvars = derivative_sys(t, vars, a, b, u, l_m, g)
    dvars(1) = -a*vars(1) + b*u(t);
    dvars(2) = -l_m*vars(2) + vars(1);
    dvars(3) = -l_m*vars(3) + u(t);
    x_hat = [vars(4), vars(5)]* [vars(2); vars(3)];
    e = vars(1) - x_hat;
    dvars(4) = g*e*vars(2);
    dvars(5) = g*e*vars(3);
    dvars = dvars';
end