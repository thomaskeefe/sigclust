function rfph = bwrfphSM(bincts,xgridp,h,deriv) 
% BWRFPHSM, Estimates Integrated Squared Density Deriv's, for 1-d k.d.e.
%   Steve Marron's matlab function
%     Binned implementation of a "diagonals in" estimate of 
%     the "Roughness of Density derivatives", (i.e. the integral of
%     their square).  The kernel is K, as opposed to K*K, which
%     comes from integrating the actual kde. For K*K plug in the
%     bandwidth * sqrt(2) for h.  
%   Assumes a Gaussian kernel.
%   Algorithm simplified from GAUSS PROC NMI2HBW.G
% Inputs:
%     bincts - column vector of bin counts (as created by gplbinr.m)
%     xgridp - vector of parameters for binning grid:
%                  [le; lr]  -  le is left end, re is right, 
%                                  of binning grid
%          h - bandwidth
%      deriv - order of derivative to calculate
% Output:
%       rfph -  Estimate of integrated squared density derivative
%

%    Copyright (c) J. S. Marron 1996-2004


%  Set initial values
lend = xgridp(1) ;
rend = xgridp(2) ;
nbin = length(bincts) ;
n = sum(bincts) ;

sh = h * ((nbin - 1) / (rend - lend)) ;
          %  Scale h to adjust for funny data scale

%  Calculate kernel function
arg = (0:(nbin - 1))' / sh ;
narg = exp(-arg.^2 / 2) / sqrt(2 * pi) ;
hp = ones(size(arg)) ;   
          %  0-th Hermite Poly
hpn = arg ;
          %  1-st Hermite Poly
for is = 1:deriv ;
  hp = arg .* hpn - (2*is - 1) * hp ;
          %  (2*is)-th Hermite Poly
  hpn = arg .* hp - (2*is) * hpn ;
          %  (2*is+1)-th Hermite Poly
end ;
kernel = hp .* narg ;



%  Get matrix of difference counts
%  (algorithm from old GAUSS PROC NMDW.G)
difwts = conv(bincts,flipud(bincts)) ;
difwts = difwts(nbin:(2*nbin-1)) ;
difwts(2:nbin) = 2 * difwts(2:nbin) ;
          %  Number of obs's for each


%  Do main calculation
rfph = difwts' * kernel ;
          %  Inclusive double sum
rfph = rfph / (n^2 * h^(2 * deriv + 1)) ;
rfph = (-1)^deriv * rfph ;

