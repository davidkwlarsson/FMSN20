function [Ex, Ez] = Estep_corrupt(theta, Y, z0, C, G, G2, B, Nsim)
% ESTEP_CORRUPT_SKELETON  Estep for corrupted data
%
% [Ex, Ez, ...] = Estep_corrupt_skeleton(theta, Y, z0, C, G, G2, B, Nsim)
%
% theta = [tau2; kappa; sigma2_epsilon; pc; beta]
% Y = column-stacked version of the observed image
% z0 = column-stacked indicator image, initial guess for corrupted pixels.
% C,G,G2 = matrices used to build a Matérn-like precision,
%          see matern_prec_matrices, sparse N-by-N
% B = the expectation basis matrix, N-by-length(beta)
% Nsim = number of simulations to use in the computations of E.
%
% [Ex, Ez, ...] = return ALL the needed expectations.
%
% This is only a skeleton for Home Assignment 3.

% $Id: Estep_corrupt_skeleton.m 4610 2012-10-26 11:59:32Z johanl $
%extract parameters
tau2 = theta(1);
kappa = theta(2);
sigma_eps = theta(3);
pc = theta(4);
beta = theta(5:end);

%compute Q
Q = [];

%compute mean
mu = [];

%set up expectations to compute
Ex = zeros(size(Q,1),1);
Ez = zeros(size(Q,1),1);
%and others...

%amd-reorder for faster computations
p = amd(Q);
C = C(p,p);
G = G(p,p);
G2 = G2(p,p);
Q = Q(p,p);
z0 = z0(p);
Y = Y(p);
mu = mu(p,:);

z = z0;
for i = 1:Nsim
  %%sample x given z and parameters
  I = find(z==0);
  A = [];
  Q_xy = [];
  mu_xy = [];
  x = [];%sample from N(mu_xy, Q_xy^-1)
  
  %%sample z given x and parameters
  P = [];%probability of being correct given y and x
  %sample new values
  z = [];%sample from the {0,1}-distribution for z.
  
  %%compute expectations given the samples
  Ex = Ex + x;
  Ez = Ez + z;
  %and others...
end
%%Compute averages
Ex = Ex/Nsim;
Ez = Ez/Nsim;
%and others...

%revert back to old order
Ex(p) = Ex;
Ez(p) = Ez;
%and others...
