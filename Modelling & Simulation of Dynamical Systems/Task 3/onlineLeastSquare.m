function theta = onlineLeastSquare(tspan, y_N, u, output_grade, input_grade, pole_filter, beta, Q0)
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
    %% Initialize method
    theta_l0 = rand(output_grade+input_grade+1, 1)*10;  %random
    P0 = inv(Q0);
    vars0 = [reshape(P0,[],1) ; theta_l0];
    %% online Least Square    
    options = odeset('RelTol',10^-8,'AbsTol',10^-9);
    Sys = @(t, vars)derivativeSys(t, vars, tspan, beta, zeta_N, y_N, size(P0));
    [~, vars] = ode45(Sys, tspan, vars0, options);
    %% Return model parameters
    theta = vars(:,numel(P0)+1:end)' + [L(2:end) , zeros(1,input_grade+1)]';

    %% Helper functions
    function dvars = derivativeSys(t, vars, tspan, beta, zeta_N, y_N, sizeP)
        %Interpolate time
        zetai = custInterpolate(zeta_N, tspan, t)';
        yi = custInterpolate(y_N, tspan, t);
        %Decode vars
        P = reshape(vars(1:sizeP(1)*sizeP(2)), sizeP);
        theta_l = vars((sizeP(1)*sizeP(2))+1:end);
        %Apply derative
        dP = beta*P - P*(zetai*zetai')*P;
        dtheta_l = P*zetai*(-zetai'*theta_l+yi);
        %Encode dvars
        dvars = [reshape(dP,[],1); dtheta_l];
    end
    function res = custInterpolate(y, t, ti)
        step = t(end)-t(end-1);
        index = floor((ti/step)+0.5)+1;  %index of time
        if (index < size(t,2))  %check if last value
            res = y(index,:) + (((y(index+1,:)-y(index,:))*(ti-t(index)))/(t(index+1)-t(index)));
        else %extend the line
            res = y(index-1,:) + (((y(index,:)-y(index-1,:))*(ti-t(index-1)))/(t(index)-t(index-1)));
        end
    end
end