clear

% Task specs
CL = 2.12; % pF
SR = 18.12; % V/us
Vdd = 1.836; % V
Vss = -Vdd; % V
GB = 7.12; % MHz
A = 20.12; % dB
P = 50.12; % mW
% Defined specs
kn = 175; % uA/V2
kp = 60; % uA/V2
Vinmax = 0.1; % V
Vinmin = -Vinmax; % V
Vt0p = -0.9056; % V
Vt0n = 0.7860; % V
Cox = 2.47; % fF/um2

% Calculate other values
% Step 1
mindim = 1; % um
% Step 2
Cc = 0.22 * CL; % pF
%Cc = 1; % pF
% Step 3
I5 = SR*Cc; % uA
% Step 4
S3 = ( I5 )/ ( kn * (Vss - Vinmin + Vt0p - Vt0n)^2);
L3 = mindim % um
if (S3 < 1)
    W3 = 1 % um
else
    W3 = S3 * L3 % um
end
S4 = S3;
L4 = L3 % um
W4 = W3 % um
% Step 5
I3 = I5 / 2; % uA
I4 = I3; % uA
gm3 = sqrt(2*kn*S3*I3); % uS
p3 = gm3 / (2*0.6667*W3*L3*Cox); % MHz
% Step 6
gm1 = GB * Cc; % uS
gm2 = gm1; % uS
S1 = ( gm1^2 ) / (kp * I5);
if (S1 < 1)
    S1 = 1;
end
L1 = mindim % um
W1 = S1 * L1 % um
L2 = L1 % um
W2 = W1 % um
% Step 7
Vsd5 = -Vinmin + Vdd - sqrt(I5/(kp * S1)) + Vt0p; % V
S5 = (2*I5)/(kp*(Vsd5^2));
if (S5 < 1)
    S5 = 1;
end
L5 = mindim % um
W5 = S5 * L5 % um
% Step 8 & 9
gm6 = 2.2 * gm1 * CL / Cc; % uS
gm4 = sqrt(2 * kn * S4 * I4); % uS
S6 = S4 * gm6 / gm4;
if (S6 < 1)
    S6 = 1;
end
L6 = mindim % um
W6 = S6 * L6 % um
I6 = ((gm6)^2)/(2*kp*S6);
% Step 10
S7 = S5 * I6 / I5;
if (S7 < 1)
    S7 = 1;
end
L7 = mindim % um
W7 = S7 * L7 % um
% Step 11
Pdiss = (I5 + I6)*(Vdd - Vss) / 1000; % mW
ln = 0.15; % 1/V
lp = 0.05; % 1/V
Av = (2 * gm2 * gm6)/(I5 * (lp + ln) * I6 *(lp + ln));
Av = 20 * log(abs(Av)); % dB
