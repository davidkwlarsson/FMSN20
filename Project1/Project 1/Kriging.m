%% Project 1 Universal Krigning
%Run this section by section
%First run the OLS script to get the residuals.

load swissRainfall.mat
Y_valid = swissRain(swissRain(:,5) == 1, :);
Y = swissRain(swissRain(:,5) == 0, :);
Y(:,1) = sqrt(Y(:,1));
Y_valid(:,1) = sqrt(Y_valid(:,1));
X_k = [ones(size(Y,1),1) Y(:,3) Y(:,2).^2];
%%

U_kk = [Y(:,3) Y(:,4)];
D_kk = distance_matrix(U_kk);

z = e_res;
Kmax = 100; 
Dmax =  max(D_kk(:))+0.001 ; 
[rhat,s2hat,m,n,d]=covest_nonparametric(U_kk,z,Kmax,Dmax);

plot(d,rhat,'LineWidth',2)
hold on 
%Assesing the uncertainty in the covariance estimate:
rhat = zeros(100,Kmax+1); 
s2hat = zeros(100,1); 
for i = 1:100
   rz = z(randperm(length(e_res))); %bootstrap
   [rhat(i,:),s2hat(i),m,n,d,varioest]=covest_nonparametric(U_kk,rz,Kmax,Dmax);
end

xlim([0 150])
y = quantile(rhat,[.025 0.975]);
plot(d,y,'--r')
par_fixed = zeros(4,1);
%Reestimates Beta. We chose the matern covariance function
[par,Beta]=covest_ml(D_kk, Y(:,1),'matern',[],X);
sigma2 = par(1); 
kappa = par(2); 
nu = par(3); 
sigma2_epsilon = par(4);
r = matern_covariance(d,sigma2,kappa,nu);
plot(d,r, 'LineWidth', 2); %Compare this to rhat above! 
legend('Covariance function, non-parametric','95% confidence interval','from the bootstrap','Covariance function, parametric')
%%
Xcoord = [Y(:,3); swissGrid(:,2)];
Ycoord = [Y(:,4); swissGrid(:,3)];
U = [Xcoord Ycoord];
D = distance_matrix(U);
Sigma = matern_covariance(D,sigma2,kappa,nu); 

%Add nugget to the covariance matrix
Sigma_yy = Sigma + sigma2_epsilon*eye(size(Sigma));

I_obs = zeros(length(Xcoord), 1); 
I_obs(1:length(Y), :) = 1;
I_obs = logical(I_obs);

%Divide Sigma_yy into observed/unobserved
Sigma_uu = Sigma_yy(~I_obs, ~I_obs);
Sigma_uk = Sigma_yy(~I_obs, I_obs);
Sigma_kk = Sigma_yy(I_obs, I_obs);

%% Validation
%plots the validation point and estimates
hold on
X_test =[ones(size(Y_valid,1),1)  Y_valid(:,3) Y_valid(:,2).^2];
plot(Y_valid(:,1),'b')
Xcoord_V = [Y(:,3); Y_valid(:,3)];
Ycoord_V = [Y(:,4); Y_valid(:,4)];
U_valid = [Xcoord_V Ycoord_V];
D_valid = distance_matrix(U_valid);
SigmaValid = matern_covariance(D_valid,sigma2,kappa,nu);
SigmaV_yy = SigmaValid + sigma2_epsilon*eye(size(SigmaValid));
I_obsV = zeros(length(Xcoord_V), 1);
I_obsV(1:length(Y), :) = 1;
I_obsV = logical(I_obsV);
SigmaV_uu = SigmaV_yy(~I_obsV, ~I_obsV);
SigmaV_uk = SigmaV_yy(~I_obsV, I_obsV);
SigmaV_kk = SigmaV_yy(I_obsV, I_obsV);
SigmaV_ku = SigmaV_uk';
Y_test = X_test*Beta + SigmaV_uk*(SigmaV_kk\(Y(:,1)-X_k*Beta));
V_muV = diag(SigmaV_uu-SigmaV_uk*(SigmaV_kk\SigmaV_ku) + ...
              ((X_test'-(X_k')*(SigmaV_kk\SigmaV_ku))')*((X_k'*(SigmaV_kk\X_k))\...
              (X_test' - X_k'*(SigmaV_kk\SigmaV_ku))));
Y_upper = Y_test + 1.96*sqrt(V_muV);
Y_lower = Y_test - 1.96*sqrt(V_muV);
plot(Y_test,'r')
plot(Y_upper,'--')
plot(Y_lower,'--')
%% Prediction
hold on
X_u = Xgrid;
Y_u = X_u*Beta + Sigma_uk*(Sigma_kk\(Y(:,1)-X_k*Beta));
mu = nan(size(swissElevation));
mu(~isnan(swissElevation))=Y_u;
imagesc([0 max(swissX(:))], [0 max(swissY(:))], mu,'alphadata', ~isnan(mu))
plot(swissBorder(:,1), swissBorder(:,2),'k')
scatter(Y(:,3), Y(:,4), 20, Y(:,1), 'filled','markeredgecolor','r')
axis xy tight; hold off; colorbar
title('sqrt of rainfall and predictions')

%% Plot the STD
hold on
STD_K = nan(size(swissElevation));
Sigma_ku = Sigma_uk';
V_mu0K = diag(Sigma_uu-Sigma_uk*(Sigma_kk\Sigma_ku) + ...
              ((X_u'-(X_k')*(Sigma_kk\Sigma_ku))')*((X_k'*(Sigma_kk\X_k))\...
              (X_u' - X_k'*(Sigma_kk\Sigma_ku))));
STD_K( ~isnan(swissElevation) ) = sqrt(V_mu0K);
imagesc([0 max(swissX(:))], [0 max(swissY(:))], STD_K,'alphadata', ~isnan(STD_K))
scatter(Y(:,3), Y(:,4), 'r')
axis xy tight; hold off; colorbar
caxis([0 0.65])
title('Standard deviation using Universal Kringing')
