function [outputArg1,outputArg2] = projectHl3(x)
%specifications resulting from the formula described
%in the pdf
cl=(2+0.01*x)*10^(-12);
sr=(18+0.01*x)*10^6;
vdd=(1.8+0.003*x);
vss=-vdd;
GB=(7+0.01*x)*10^6;
a=(20+0.01*x);
p=(50+0.01*x)*10^(-3);

%calculating design parameters with L=10u, all parameters 
%are in SI units

L=0.7*10^(-6);
kp=60*10^(-6);
kn=175*10^(-6);
ln=0.05;
lp=0.15;
VinMax=100*10^(-3);
VinMin=-VinMax;
Cgsp=4.0241*10^(-10);
Cgsn=5.3752*10^(-11);

Vtn=0.7860;  %values used for these two parameters ignore 
Vtp=-0.9056; %body effect present


%-----------------parameters-------------------%

%-----------calculate Cc compensation capacitor-------------------------
Cc=4*cl; %for phase margin 60

%-------------calculate I5 to accomodate for the sr specification ------
I5=sr*Cc; 

%-----------------S3 for minimum input specification--------------------

%check=1;

%while(check)
S3=I5/(kn*(-vss+VinMin-abs(Vtp)+Vtn)^2);
if(S3<1)
    W3=10^-(6);
else
    W3=S3*L;
end

%--------------check that p3 is greater than 10GB----------------------
%--------------if not then increase I5 recalculate S3------------------
%------------------------and recheck-----------------------------------
gm3=sqrt(abs(I5*kn*S3));

%if(gm3>10*GB*2*Cgsn)
    %check=0;
%else
    %I5=I5+10^(-6);
%end

%end

%---------------calculate S1 and S2------------------------------------
S1=(GB*Cc)^2/(kp*I5);
W1=S1*L;
gm1=GB*Cc;

%-------calculate Vov5 and S5 to achieve maximuum input specification--
Vov5=vdd-abs(Vtp)-sqrt(I5/(kp*S1));
S5=2*I5/(kp*Vov5^2);
W5=S5*L;

%-------------calculate S6 in order to achieve best mirroring----------
%----------------------at input stage----------------------------------
gm6=40*gm1*(cl/Cc);
S6=S3*gm6/gm3;
W6=S6*L;
I6=gm6^2/(2*kn*S6);

%-------------calculate S7 in order to eliminate zero input output-----
%------------------------------due to second stage---------------------
S7=S5*(I6/I5);
W7=S7*L;

%-------------calculating expected gain and Pdiss----------------------
A=2*gm1*gm6/(I5*(ln+lp)^2*I6);
Av=20*log10(abs(A));
Pdiss=(I5+I6)*2*vdd;

%printing results
fprintf('Cc= %d\nI5= %d\nW3= %d\nW1-W2= %d\nW5= %d\nVov5= %d\n',Cc,I5,W3,W1,W5,Vov5);
fprintf('W6-I6= %d-%d\nW7= %d\nAv= %d\nPdiss= %d\n',W6,I6,W7,Av,Pdiss);



end

