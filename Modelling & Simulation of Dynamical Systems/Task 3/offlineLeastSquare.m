function theta = offlineLeastSquare(tspan, y_N, u, output_grade, input_grade, pole_filter)
    %% Build characteristic equation of the filetr
    L = poly(ones(1,output_grade)*pole_filter);
    %% Linearize model
    zeta_N = NaN(size(y_N,1), output_grade+input_grade+1);
    for i = 1:output_grade+input_grade+1
        if i <=output_grade    %for output
            %build zeta_one element
            tmp = zeros(1, output_grade+1-i);
            tmp(1) = -1;
            zeta = tf(tmp,L);
            %simulate that element
            zeta_N(:,i) = lsim(zeta, y_N, tspan);
        else        %for input
            %build zeta_two element
            tmp = zeros(1, input_grade+1+output_grade+1-i);
            tmp(1) = 1;
            zeta = tf(tmp,L);
            %simulate that element
            zeta_N(:,i) = lsim(zeta, u, tspan);
        end
    end
    %% Least Square
    theta_l = linsolve(zeta_N'*zeta_N , zeta_N'*y_N);
    %% Return model parameters
    theta = theta_l + [L(2:end) , zeros(1,input_grade+1)]';
end