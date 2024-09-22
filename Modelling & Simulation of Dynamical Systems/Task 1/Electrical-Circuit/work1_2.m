clc;
clear;

time_s = 0.0;    %sec
tstep = 0.00001;   %sec
time_f = 10;   %sec
tspam = time_s:tstep:time_f;

% Sample input-output
samplesize = size(tspam, 2);
index = 0;
VR_N = zeros(samplesize, 1);  %initialize memory
VC_N = zeros(samplesize, 1);
u1_N = zeros(samplesize, 1);
u2_N = zeros(samplesize, 1);
for t = tspam
    Vout = v(t);
    index = index + 1;
    VR_N(index) = Vout(2);    %V
    VC_N(index) = Vout(1);    %V    
    u1_N(index) = 2*sin(t);   %V
    u2_N(index) = 1;          %V
end

% Import error (uncomment to apply)
%[VR_N , VC_N] = error_import(VR_N, VC_N, samplesize);

% Linear Model
fpoles = 100;   %filter poles
L = tf([1 2*fpoles fpoles*fpoles], 1);  %L(s) = (s + fpoles)^2

%1.Method with VR
y_vr = [tf([1 0 0],1)/L , tf([1 0 0],1)/L , tf([-1 0 0],1)/L];
zeta_vr = [tf([1 0],1)/L , 1/L, -1/L];
%Least Square
y_vr_N = lsim(y_vr(1), u1_N, tspam) + lsim(y_vr(2), u2_N, tspam) + lsim(y_vr(3), VR_N, tspam);
zeta_vr_N = [ lsim(zeta_vr(1), VR_N, tspam) , lsim(zeta_vr(2), VR_N, tspam) , lsim(zeta_vr(3), u1_N, tspam) ];
theta_l_vr_zero = linsolve(zeta_vr_N'*zeta_vr_N , zeta_vr_N'*y_vr_N);

%2.Method with VC
y_vc = tf([1 0 0],1)/L;
zeta_vc = [ tf([-1 0],1)/L , -1/L , tf([1 0],1)/L , tf([1 0],1)/L , 1/L ];
%Least Square
y_vc_N = lsim(y_vc, VC_N, tspam);
zeta_vc_N = [ lsim(zeta_vc(1), VC_N, tspam) , lsim(zeta_vc(2), VC_N, tspam), ...
              lsim(zeta_vc(3), u1_N, tspam) , lsim(zeta_vc(4), u2_N, tspam) , lsim(zeta_vc(5), u2_N, tspam) ];
theta_l_vc_zero = linsolve(zeta_vc_N'*zeta_vc_N , zeta_vc_N'*y_vc_N);

% Find method error
%1. for VR
RC_vr_m = theta_l_vr_zero(1); % 1/RC from model
LC_vr_m = (theta_l_vr_zero(2) + theta_l_vr_zero(3)) / 2; % 1/LC from model
VR_N_m = lsim( tf([1 0 LC_vr_m], [1 RC_vr_m LC_vr_m]), u1_N, tspam) ...
        + lsim( tf([1 0 0], [1 RC_vr_m LC_vr_m]), u2_N, tspam);
%2. for VC
RC_vc_m = (theta_l_vc_zero(1) + theta_l_vc_zero(3) + theta_l_vc_zero(4)) / 3; % 1/RC from model
LC_vc_m = (theta_l_vc_zero(2) + theta_l_vc_zero(5)) / 2; % 1/LC from model
VC_N_m = lsim( tf([RC_vc_m 0], [1 RC_vc_m LC_vc_m]), u1_N, tspam) ...
        + lsim( tf([RC_vc_m LC_vc_m], [1 RC_vc_m LC_vc_m]), u2_N, tspam);
VR_error = mean(abs(VR_N - VR_N_m));
VC_error = mean(abs(VC_N - VC_N_m));
display(VR_error);
display(VC_error);

function [VR_N, VC_N] = error_import(VR_N, VC_N, samplesize)
    for i = 1:3
        rand_index = randi(samplesize); % pick random value
        VR_N(rand_index) = rand() * power(10, randi([0 4]));    % add error from 0 to 10^4
        rand_index = randi(samplesize); % pick random value
        VC_N(rand_index) = rand() * power(10, randi([0 4]));    % add error from 0 to 10^4
    end
end