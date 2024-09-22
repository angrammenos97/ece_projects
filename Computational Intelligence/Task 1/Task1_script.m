%% Computational Intelligence Task 1
% Anastasios Gramemnos    9212
% avgramme@ece.auth.gr    March 2022

%% Define open loop transfer function
clear
s = tf('s');
Gp = 10/((s+1)*(s+9));

%% Design linear controller
Kp = 1.5;
Ki = 2.25;
c = Ki/Kp;
Gc = Kp*(s+c)/s;
system_open_loop = Gc*Gp;
system_closed_loop = feedback(system_open_loop, 1, -1);
step(system_closed_loop);
disp(stepinfo(system_closed_loop));

%% Design FLC
FLC = readfis("FLC.fis");
gensurf(FLC);
load_system('Variable_Input');
open_system('Variable_Input/Scope');
simout = sim('Variable_Input');