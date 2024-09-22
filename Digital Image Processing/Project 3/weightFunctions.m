function w=weightFunctions(weightingFcn, Zmin, Zmax, t)
  wUniform  = @(z)    and(z>=Zmin, z<=Zmax).*(Zmax/Zmax); 
  wTent     = @(z)    and(z>=Zmin, z<=Zmax).*(min(z-Zmin,Zmax-z)./Zmax);
  wGaussian = @(z)    and(z>=Zmin, z<=Zmax).*(exp(-4*((z-(Zmax-Zmin)).^2)/((Zmax-Zmin)^2)));
  wPhoton   = @(z)    and(z>=Zmin, z<=Zmax).*t;

  switch weightingFcn
    case 'uniform'
      w = wUniform;
    case 'tent'
      w = wTent;
    case 'gaussian'
      w = wGaussian;
    otherwise
      w = wPhoton;
  end
end