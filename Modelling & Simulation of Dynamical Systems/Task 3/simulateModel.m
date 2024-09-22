function y = simulateModel(theta, u, tspan, output_grade)
    P = theta(output_grade+1:end)';
    Q = [1 ,theta(1:output_grade)'];
    H = tf(P, Q);
    y = lsim(H, u, tspan);
end