function [gibbs,corrupt] = Gibbs_param(theta, mu_x,C, G, G2, img, Nsim, corrupt,u1,u2)

%theta contains
tau = theta(1);
kappa2 = theta(2);
sigma2 = theta(3);
pc = theta(4);
mu_x = mu_x;
Q = (kappa2^2*C + 2*kappa2*G + G2);
[m,n] = size(img);
img_col = img(:);
N = length(img_col);
tau_all = zeros(Nsim,1);
kappa_all = zeros(Nsim,1);
sigma2_all = zeros(Nsim,1);
pc_all = zeros(Nsim,1);
% known = corrupt(:);

known = rand(N,1) < pc;
Y = img_col(known);
xmiss = img_col.*known;


for i=1:Nsim
    
    Qx = tau*Q;
    A = sparse(1:length(Y), find(known), 1, length(Y), numel(xmiss));
    Aall = [A ones(length(Y), 1)];
    Qbeta = 1e-6 * speye(1);
    Qall = blkdiag(Qx,Qbeta);
    Qeps = speye(length(Y))/sigma2;
    Qxy = Qall + Aall'*Qeps*Aall;
    p = amd(Qxy);
    Qxy = Qxy(p,p);
    Aall = Aall(:,p);
    Rxy = chol(Qxy);
    E_xy = Rxy\((Aall'*Qeps*Y)'/Rxy)';
    x_samp = E_xy + Rxy\randn(size(Rxy,1),1);
    x_samp(p) = x_samp;
    z_samp = [speye(size(Q,1)) ones(size(Q,1),1)]*x_samp;
    
    N = length(img_col);
    K = 2;
    p = zeros(N,K); 
    y = img_col-z_samp;
    p(:,1) = exp(-0.5*(y.^2/sigma2)) / ...
           ((2*pi)^(1/2)*sigma2^(0.5));
    p(:,2) = 1;
    prior = [pc, 1-pc];
    p = p*diag(prior);
    p(:,1) = p(:,1)./sum(p,2);
    
    im_rnd = rand(N,1);
    
    known = p(:,1) > im_rnd;
    n_obs = sum(known);
    n_tot = length(known);
    xmiss = img_col.*known;
    Y=img_col(known);
    
%     n_obs = sum(known);
%     n_tot = length(img_col);
    pc = betarnd(n_obs + 1,n_tot-n_obs + 1);
%     Y = img_col(known);
%     xmiss = img_col.*known;
    
    shape = N/2 + 1;
    scale = 2/(z_samp'*Q*z_samp);
    tau = gamrnd(shape, scale);
    tau_all(i,1) = tau;
    N = length(Y);
    sigma2_inv = gamrnd(N/2+1,2/sum((Y-z_samp(known)).^2));
    sigma2 = 1/sigma2_inv;
    
    kappa_all(i,1) = kappa2;
    sigma2_all(i,1) = sigma2;
    pc_all(i,1) = pc;
    subplot(221)
    plot(tau_all)
    subplot(222)
    plot(kappa_all)
    subplot(223)
    plot(sigma2_all)
    subplot(224)
    plot(pc_all)
    drawnow()
  
    
end

gibbs = [tau, kappa2, sigma2, pc];

end