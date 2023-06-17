var y R pi z g YGR INFL INT;
varexo eg ep ez;

parameters tau rho_R psi1 psi2 rho_g rho_z kappa  piA gammaQ rA s_ep s_eg s_ez; 

rA = 0.42;
piA = 3.3;
gammaQ = 0.52;
tau = 2.83;
kappa = 0.78;
psi1 = 1.8;
psi2 = 0.63;
rho_R = 0.77;
rho_g = 0.98;
rho_z = 0.88;
s_ep = 0.22;
s_eg = 0.72;
s_ez = 0.31;


model;
#  beta = 1/(1 + rA/400);
  y = y(+1) - (1/tau)*(R - pi(+1) - z(+1)) + g - g(+1);
  pi = beta*pi(+1) + kappa*(y - g);
  R = rho_R*R(-1) + (1 - rho_R)*(psi1*pi + psi2*(y - g)) + s_ep*ep/100;
  g = rho_g*g(-1) + s_eg*eg/100;
  z = rho_z*z(-1) + s_ez*ez/100;
  YGR = gammaQ + 100*(y - y(-1) + z);
  INFL = piA + 400*pi;
  INT = piA + rA + 4*gammaQ + 400*R;
end;

steady_state_model;
  y = 0;
  pi = 0;
  R = 0;
  g = 0;
  z = 0;
  YGR = gammaQ;
  INFL = piA;
  INT = piA + rA + 4*gammaQ;
end;

shocks;
  var ep; stderr 1.0;
  var eg; stderr 1.0;
  var ez; stderr 1.0;
  // calibrated measurement errors
  var YGR;  stderr (0.20*0.579923);
  var INFL; stderr (0.20*1.470832);
  var INT;  stderr(0.20*2.237937);
end;

rA.prior(shape=gamma,mean=0.5,stdev=0.5);
piA.prior(shape=gamma,mean=7,stdev=2);
gammaQ.prior(shape=normal, mean=0.4, stdev=0.2);
tau.prior(shape=gamma, mean=2, stdev=0.5);
kappa.prior(shape=uniform, mean=0.5, variance = 1/12, domain=[0,1]);
psi1.prior(shape=gamma, mean=1.5, stdev=0.25);
psi2.prior(shape=gamma, mean=0.5, stdev=0.25);
rho_R.prior(shape=uniform, mean=0.5, variance=1/12, domain=[0,1]);
rho_g.prior(shape=uniform, mean=0.5, variance=1/12, domain=[0,1]);
rho_z.prior(shape=uniform, mean=0.5, variance=1/12, domain=[0,1]);
// s=0.4, nu=4
//s_ep.prior(shape=inv_gamma1, mean=0.32, stdev=1000);
s_ep.prior(shape=inv_gamma1, mean=0.5013256549262002, variance=0.06867258771281654);
// s=1, nu=4
s_eg.prior(shape=inv_gamma1, mean = 1.2533141373155003, variance=0.4292036732051032);
// s=0.5, nu=4
s_ez.prior(shape=inv_gamma1, mean = 0.6266570686577502, variance=0.1073009183012758);

//stoch_simul(order=1, irf=0);

estimated_params_init;
  tau, 2.263895645223946;
  kappa, 0.999999883197828;
  psi1, 1.932357745633903;
  psi2, 0.465714640873466;
  rA, 0.333453026636814;
  piA, 3.427852381637088;
  gammaQ, 0.624767199308368;
  rho_R, 0.764476492352786;
  rho_g, 0.990359718181570;
  rho_z, 0.917302546854851;
  s_ep,   0.211571288023659;
  s_eg, 0.632177304244290;
  s_ez, 0.192160214260411;
end;

varobs YGR INFL INT;

//estimation(datafile='dsge1_data.csv', mode_compute=0, mh_replic=100000);
