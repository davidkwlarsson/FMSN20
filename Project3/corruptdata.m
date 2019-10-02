%% Home Assignment 3: Corrupted Data
titan = imread('titan.jpg');
img=double(titan)/255;
% img = imnoise(x,'gaussian');
x = img;
imagesc(x);
colormap(gray);
[m,n] = size(x);
x0 = rand(m,n);
pc = 0.5;
corrupted_index = x0<pc; %ones are not corrupted.
unif = rand(m,n);
x(~corrupted_index) = unif(~corrupted_index);
imagesc(x);

%% Gibbs Algorithm:

[u1,u2] = ndgrid(1:m,1:n);
[C,G,G2] = matern_prec_matrices([u1(:),u2(:)]);
kappa2 = 0.1;
tau = 1;
sigma2 = 0.1;
pc = 0.5;
%%
Qx = tau*(kappa2^2*C + 2*kappa2*G + G2);
xm = x(:);
mu_x = mean(xm);

%%
kappa2 = 0.01;
tau = 500;
sigma2 = 0001;
pc = 0.5;
Nsim = 10;
theta = [tau; kappa2; sigma2; pc];
%%
theta_new = Gibbs(theta,C,G,G2,x,Nsim);
%%
tau = theta_new(1);
sigma2 = theta_new(3);
pc = theta_new(4);



