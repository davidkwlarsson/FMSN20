function [gibbs,corrupt,im] = Gibbs(theta,C, G, G2, img, Nsim)
%theta contains
tau = theta(1);
kappa2 = theta(2);
sigma2 = theta(3);
pc = theta(4);
Q = (kappa2^2*C + 2*kappa2*G + G2);
p = amd(G2);
% C = C(p,p);
% G = G(p,p);
% G2 = G2(p,p);
% Q = Q(p,p);
[m,n] = size(img);
img_col = img(:);
N = length(img_col);
x = img_col;

% theta = [tau; kappa2; sigma2_epsilon; pc];
tau_all = zeros(Nsim,1);
kappa_all = zeros(Nsim,1);
sigma2_all = zeros(Nsim,1);
pc_all = zeros(Nsim,1);
sum_known= zeros(N,1);
burnin = Nsim-10; 

mean_img = zeros(N,10);
K = 2;
known = rand(N,1) < pc;
    xmiss = x.*known;
    Y=x(known);

for i=1:Nsim
    N = length(img_col);
    % create field
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
    E_xy = Rxy\((Aall'*Y/sigma2)'/Rxy)';
    
    x_samp = E_xy + Rxy\randn(size(Rxy,1),1);
    x_samp(p) = x_samp;
    E_xy(p) = E_xy;
    z = [speye(size(Q,1)) ones(size(Q,1),1)]*x_samp;
    
    % probabilities for each pixel
    p = zeros(N,K); 
    p(:,1) = exp(-0.5*((x-z).^2/sigma2)) / ...
           ((2*pi)^(1/2)*sigma2^(0.5));
    p(:,2) = 1;
    prior = [pc, 1-pc];
    
    p = p*diag(prior);
    norm = sum(p,2); 
    p(:,1) = p(:,1)./norm;
    
%     imagesc(reshape(p(:,1),[m,n]));
%     title(num2str(pc))
%     pause(0.5)
    known = p(:,1) > rand(N,1);
    n_obs = sum(known);
    n_tot = length(known);
    pc = betarnd(n_obs + 1,n_tot-n_obs + 1);
    
    % tau
    N = length(Qx);
    shape = N/2 + 1; % double check this sign
    scale = 2/(x_samp(1:end-1)'*Q*x_samp(1:end-1)); % x eller E_zy eller E_xy
    tau = gamrnd(shape, scale);
    tau_all(i,1) = tau;
    
    %sigma
    N = sum(known);
    b = x(known)-z(known);
    sigma2_inv = gamrnd(N/2+1,2/sum(b.^2));
    sigma2 = 1./sigma2_inv;
    
    kappa_all(i,1) = kappa2;
    sigma2_all(i,1) = sigma2;
    pc_all(i,1) = pc;
    
    xmiss = x.*known;
    Y=x(known);
    if i > burnin 
        sum_known = sum_known + known; 
        mean_img(:,i-burnin) = z; 
    end
end

im = mean(mean_img,2);
corrupt = sum_known./(Nsim-burnin); 
subplot(221)
plot(tau_all)
title('tau')
subplot(222)
plot(kappa_all)
subplot(223)
plot(sigma2_all)
title('sigma')
subplot(224)
plot(pc_all)
title('pc')
figure(); 

imagesc(reshape(im,[m,n]))

gibbs = [tau, kappa2, sigma2, pc]
end