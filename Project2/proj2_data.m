%load data
load ('HA2_forest')

subplot(221)
imagesc( bei_counts )
subplot(222)
imagesc( isnan(bei_counts) )
subplot(223)
imagesc( bei_elev )
subplot(224)
imagesc( bei_grad )

%size of the grid
sz = size(bei_counts);
%observations
Y = bei_counts(:);

%% Create modeling data and validation data
IndexV = 1:5000;
IndexV = IndexV';
IndexM = reshape(IndexV, sz);
index = [577 3450 4965 558 3958 4478 508 158 2977 769 1734 2262 928 2696];
Yvalid = [Y(index) index'];
Y(index) = NaN;

%% Only change this for different models

%missing data
I = ~isnan(Y);
%create Q-matrix
[u1, u2] = ndgrid(1:sz(1),1:sz(2));
[C,G,G2] = matern_prec_matrices([u1(:) u2(:)]);
%mean value-vector (might not need all)
Bgrid = [ones(prod(sz),1) bei_elev(:) bei_grad(:)];
%and observation matrix for the grid
Agrid = speye(prod(sz));

%G2 is the most dense of the matrices, lets reorder
p = amd(G2);
figure
subplot(121)
spy(G2)
subplot(122)
spy(G2(p,p))

%reorder precision matrices
C = C(p,p);
G = G(p,p);
G2 = G2(p,p);
%and observation matrix
Agrid = Agrid(:,p);

%create A tilde matrix
Atilde = [Agrid Bgrid]; 

%we need a global variable for x_mode to reuse
%between optimisation calls
global x_mode;
x_mode = [];

%subset Y and Atilde to observed points
isCar = true;
par = fminsearch( @(x) GMRF_negloglike(x, Y(I), Atilde(I,:), C, G, G2, 1e-6, isCar), [0 0]);
%conditional mean is given by the mode  
E_xy = x_mode;
%and reconstruction (field+covariates)
%%
Em_zy = Bgrid*x_mode(size(Agrid,2)+1:end);
Es_zy = Agrid*x_mode(1:size(Agrid,2));
Ec_zy = Atilde*x_mode;
subplot(311)
imagesc( reshape(Em_zy,sz) )
caxis([-5 5])
subplot(312)
imagesc( reshape(Es_zy,sz) )
caxis([-5 5])
subplot(313)
imagesc( reshape(Ec_zy,sz) )
caxis([-5 5])
%%
tau = exp(par(1));
kappa2 = exp(par(2));
if isCar
    Q_x = tau*(kappa2*C+G);
else
    Q_x = tau*(kappa2^2*C + 2*kappa2*G + G2); 
end
Nbeta = size(Atilde,2) - size(Q_x,1);
Qtilde = blkdiag(Q_x, 1e-6*speye(Nbeta));
%reuse taylor expansion to compute posterior precision
[~, ~, Q_xy] = GMRF_taylor(E_xy, Y(I), Atilde(I,:), Qtilde);

%1000 samples from the approximate posterior
Rxy = chol(Q_xy);
x_samp = bsxfun(@plus, E_xy, Rxy\randn(size(Rxy,1),1000));


%%
z_samp_mean = Bgrid*x_samp(length(Agrid)+1:end,:);
z_samp_spacial = Agrid*x_samp(1:length(Agrid),:);
z_samp_complete = Atilde*x_samp;
y_samp = exp(z_samp_complete);
y_std = std(y_samp,0,2);

std_z_mean = std(z_samp_mean,0,2);
std_z_spacial = std(z_samp_spacial,0,2);
std_z_complete = std(z_samp_complete,0,2);
subplot(131)
imagesc( reshape(std_z_mean,sz))
title('Mean')
subplot(132)
imagesc( reshape(std_z_spacial,sz))
title('Spacial')
subplot(133)
imagesc( reshape(std_z_complete,sz))
title('Complete')

e = [zeros(size(Q_xy,1)-size(Bgrid,2), size(Bgrid,2)); eye(size(Bgrid,2))];
V_beta0 = e'*(Q_xy\e); % check so that the covariates actually are significant..

%fält och intercept starkt korrelerat
figure();plot(mean(x_samp(1:5000,:)),x_samp(5001,:),'.')

%%  check so that the covariates actually are significant..
beta = E_xy(end-size(Bgrid,2)+1:end);
conf = 2*sqrt(diag(V_beta0));
beta_min = beta-conf;
beta_max = beta+conf;

%% Validation plot
figure
hold on
plot(Yvalid(:,1),'r')
plot(exp(Ec_zy(Yvalid(:,2))),'b')
plot(exp(Ec_zy(Yvalid(:,2)))-1.96*sqrt(exp(Ec_zy(Yvalid(:,2)))),'b--')
plot(exp(Ec_zy(Yvalid(:,2)))+1.96*sqrt(exp(Ec_zy(Yvalid(:,2)))),'b--')
legend({'validation','estimates','prediction intervall'},'Location','southwest')
error = Yvalid(:,1)-exp(Ec_zy(Yvalid(:,2)));
if isCar 
    MSE_CAR = mean(error.^2);
else
    MSE_SAR = mean(error.^2);
end
%%
subplot(221)
imagesc( log(bei_counts) )
subplot(222)
imagesc( reshape(Em_zy,sz) )
subplot(223)
imagesc( reshape(Ec_zy,sz))
subplot(224)
imagesc( reshape(Es_zy,sz) )
