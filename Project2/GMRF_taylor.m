function [g, dg, d2g]= GMRF_taylor(x_0, y, A, Q)
% GMRF_TAYLOR_SKELETON  Taylor expansion of the conditional for non-Gaussian observations
%
% [f, df, d2f]= GMRF_taylor_skeleton(x_0, y, A, Q)
%
% x_0 = value at which to compute taylor expansion
% y = the data vector, as a column with n elements
% A = the observation matrix, sparse n-by-N
% Q = the precision matrix, sparse N-by-N
%
% Function should return taylor expansion of
%   g = -log p(y|x) + 1/2 x'*Q*x = -f + 1/2 x'*Q*x
% as well as gradient and Hessian. Sign shift since matlab does
% minimisation instead of maximisation
%
% This is only a skeleton for Home Assignment 2.

% $Id: gmrf_negloglike_skeleton.m 4454 2011-10-02 18:29:12Z johanl $

%compute log observations, and derivatives
z = A*x_0;

logp = y.*z - exp(z);% this is a constant, -sum(log(y)) % log(p(yi|zi,theta) 
%compute the function
g = x_0'*Q*x_0/2 - sum(logp);

if nargout>1
  %compute derivatives (if needed, i.e. nargout>1)
  d_logp = y - exp(z);
  dg = Q*x_0 - A'*d_logp;
end

if nargout>2
  %compute hessian (if needed, i.e. nargout>2)
  d2_logp =- exp(z); 
  n = size(A,1);
  d2g = Q - A'*spdiags(d2_logp,0,n,n)*A;
end