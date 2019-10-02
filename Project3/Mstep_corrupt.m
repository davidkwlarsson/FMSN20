function theta_est = Mstep_corrupt(theta, Y, C, G, G2, B, Ex, Ez)
% MSTEP_CORRUPT_SKELETON  Mstep for corrupted data
%
% [Ex, Ez, ...] = Mstep_corrupt_skeleton(theta, Y, C, G, G2, B, Ex, Ez, ...)
%
% theta = [tau2; kappa; sigma2_epsilon; pc; beta]
% Y = column-stacked version of the observed image
% C,G,G2 = matrices used to build a Matérn-like precision,
%          see matern_prec_matrices, sparse N-by-N
% B = the expectation basis matrix, N-by-length(beta)
% Ex, Ez, ... = Expectations computed by Estep_corrupt_skeleton
%
% This is only a skeleton for Home Assignment 3.

% $Id: Mstep_corrupt_skeleton.m 4586 2012-10-08 16:18:33Z johanl $

%First set parameters equal to the old parameters.
theta_est = theta;

%%estimate tau --- assuming fixed range (keep kappa)
theta_est(1) = [];

%%estimate the measurement variance
theta_est(3) = [];

%%average error probability
theta_est(4) = [];

%%estimate beta
beta = [];
theta_est(5:end) = beta;

