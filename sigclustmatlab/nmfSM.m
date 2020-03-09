function vf = nmfSM(xgrid,vmu,vsig2,vw) 
% NMF, Normal Mixture density Function
%   Steve Marron's matlab function
%     Evaluates normal mixture probability density function at xgrid
%   Can use 2 or 4 arguments.
% Inputs:
%     xgrid - Column vector of points at which to evaluate density
%       vmu - Column vector of means
%             or if no other arguments, a 3 column matrix of:
%                  means, variances and weights
%     vsig2 - Column vector of variances
%        vw - Column vector of weights (should sum to 1)
% Output:
%        vf - Column vector of density heights
%
% Assumes path can find personal function:
%    vec2matSM.m

%    Copyright (c) J. S. Marron 1996-2001


%  Set parameters according to number of input arguments
%
if nargin == 2 ;       %  only 1 argument input, use columns as params
  w = vmu(:,3) ;
  sig2 = vmu(:,2) ;
  mu = vmu(:,1) ;
elseif nargin == 4 ;   %  then parameter vector vectors input separately
  mu = vmu ;
  sig2 = vsig2 ;
  w = vw ;
end ;


nxgrid = length(xgrid) ;

if nxgrid <= 1 ;
  if length(mu) <= 1 ;    %  Then do all scalar operations
    vf = xgrid - mu ;
    vf = exp(-(vf.^2) ./ (2 * sig2)) ;
    vf = vf ./ sqrt(2 * pi * sig2) ;
    vf = vf .* w ;
  else ;        %  Then have just one x, but need to sum components
    vf = xgrid * ones(length(mu),1) - mu ;
    vf = exp(-(vf.^2) ./ (2 * sig2)) ;
    vf = vf ./ sqrt(2 * pi * sig2) ;
    vf = vf .* w ;
    vf = sum(vf)' ;
  end ;
else ;
  if length(mu) <= 1 ;    %  Then have just one component, don't sum
    vf = xgrid' - mu * ones(1,nxgrid) ;
    vf = exp(-(vf.^2) ./ (2 * sig2 * ones(1,nxgrid))) ;
    vf = vf ./ (sqrt(2 * pi * sig2) * ones(1,nxgrid)) ;
    vf = vf * w ;
    vf = vf' ;
  else ;        %  Then have full matrices for operations
    vf = vec2matSM(xgrid',length(mu)) - vec2matSM(mu,nxgrid) ;
    vf = exp(-(vf.^2) ./ (2 * vec2matSM(sig2,nxgrid))) ;
    vf = vf ./ vec2matSM(sqrt(2 * pi * sig2),nxgrid) ;
    vf = vf .* vec2matSM(w,nxgrid) ;
    vf = sum(vf)' ;
  end ;
end ;
  
