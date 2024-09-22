function dstates = dynamics(t, states , b, k, m)

u = 5*sin(2*t) + 10.5;  %N

dstates(1) = states(2);
dstates(2) = -(k/m)*states(1) -(b/m)*states(2) +u;
dstates = dstates';

end
