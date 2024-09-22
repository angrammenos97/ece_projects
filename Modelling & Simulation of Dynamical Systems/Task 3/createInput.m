function [u, t] = createInput(time, max_amplitude, max_angular_velocity, number_of_sin)
    if (numel(time) == 2)
        t = 0:time(1):time(2);
    else
        t = time;
    end
    u = zeros(size(t,1), 1);
    for n = 1:number_of_sin
       amp_i = rand() * max_amplitude;
       w_i = rand() * max_angular_velocity;
       if rand() > 0.5  %sin
           u = u + (amp_i*sin(w_i*t));
       else             %cos
           u = u + (amp_i*cos(w_i*t));
       end
    end
end