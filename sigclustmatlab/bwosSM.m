function hos = bwosSM(data) 
% BWOSSM, Terrell's OverSmoothed BandWidth, for 1-d kernel density estimation
%   Steve Marron's matlab function
%     This is the bandwidth, which maximizes the MISE asymptotically
%     optimal representation, modulo "scale", chosen here to
%     be "standard deviation".
%     Assumes the kernel is standard Gaussian
% Inputs:
%     data - column vector of data,
%              or else 1 x 2 row vector, where:
%                     data(1) = nobs
%                     data(2) = standard deviation
%                 (useful when data are binned)
% Output:
%     hos - S.D. based oversmoothed bandwidth
%

%    Copyright (c) J. S. Marron 1996-2004

if size(data,2) == 1 ;        %  if a column vector
  n = length(data) ;
  dsd = std(data) ;
elseif size(data,1) == 1 ;    %  if a row vector
  n = data(1) ;
  dsd = data(2) ;
end ;

  rk = 1 / (2 * sqrt(pi)) ;
          %  Integral of the square of the Gaussian kernel
  osc = 3 * 35^(-1/5) ;
          %  Oversmoothing constant = R(f''))(-1/5), for special f
c = osc * rk^(1/5) ;

hos = c * dsd * n^(-1/5) ;

