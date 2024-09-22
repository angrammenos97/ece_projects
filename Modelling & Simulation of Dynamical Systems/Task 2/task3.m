clear;
clc;

% Specifications
A = [ -0.25, 3 ; -5, -1 ];
B = [ 1 ; 2.2 ];
u = @(t) 10*sin(2*t) + 5*sin(7.5*t);
time_s = 0.0;   %start time
time_f = 50.0;  %end time

% Initialize
X0 = [ 0 ; 0];
A_hat0 = [ 1, 2 ; 3, 4 ];   %random
B_hat0 = [ 5 ; 6 ];         %random
vars0 = [X0 ; X0 ; reshape(A_hat0,[],1); B_hat0];

% Lyapunov method
tspan = [time_s time_f];
options = odeset('RelTol',10^-10,'AbsTol',10^-11);
Sys = @(t, vars)derivative_sys(t, vars, A, B, u);
[t, vars] = ode15s(Sys, tspan, vars0, options);

% Visualize
figure(1);
plot(t, vars(:,3), t, vars(:,1));
legend("x1-hat", "x1 real");
figure(2);
plot(t, vars(:,4), t, vars(:,2));
legend("x2-hat", "x2 real");
figure(3);
plot(t, vars(:,5:8));
legend("a11-hat", "a21-hat", "a12-hat", "a22-hat");
figure(4);
plot(t, vars(:,9:10));
legend("b1-hat", "b2-hat");

function dvars = derivative_sys(t, vars, A, B, u)
    X = [vars(1);vars(2)];
    X_hat = [vars(3);vars(4)];
    A_hat = [vars(5),vars(7);vars(6),vars(8)];
    B_hat = [vars(9);vars(10)];
    
    dX = A*X + B*u(t);
    dX_hat = A_hat*X_hat + B_hat*u(t);
    e = X - X_hat;
    dA_hat = e*X_hat';
    dB_hat = e*u(t);
    
    dvars = [dX ; dX_hat ; reshape(dA_hat,[],1); dB_hat];
end