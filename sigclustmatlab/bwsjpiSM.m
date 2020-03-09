function hsjpi = bwsjpiSM(data,xgridp,hgridp,imptyp,eptflag) 
% BWSJPISM, Sheather Jones Plug In (Binned version), for 1-d k.d.e.
%   Steve Marron's matlab function
%     Does data-based bandwidth selection for 1-d kernel density 
%     estimation, using the Sheather Jones Plug In method.
%     A binned implementation is used, and a grid search is done, 
%     taking the largest local downcrossing.
%   Assumes a Gaussian kernel.
%   Algorithm simplified from GAUSS PROC NMSEGBW.G
%   Can use first 1, 2, 3, 4 or 5 arguments.
% Inputs:
%     data   - either n x 1 column vector of 1-d data
%                or vector of bincts, when imptyp = -1
%     xgridp - vector of parameters for binning grid:
%                  0 (or not specified)  -  use endpts of data and 401 bins
%                  [le; lr]  -  le is left end, re is right, 401 bins
%                         (get error message and no return if le > lr)
%                  [le; lr; nb] - le left, re right, and nb bins
%     hgridp - vector of parameters for bandwidth grid:
%                  0 (or not specified)  -  use [hos/9; hos; 21]
%                  [hmin; hmax]  -  hmin is left end, hmax is right, 
%                         and use 21 logarithmically spaced bandwidths
%                         (get error message and no return if hmin > hmax)
%                  [hmin; hmax; nh] - hmin left, hmax right, and nh h's
%    imptyp  - flag indicating implementation type:
%                 -1  -  binned version, and "data" is assumed to be
%                                   bincounts of prebinned data
%                  0 (or not specified)  -  linear binned version
%                                   and bin data here
%    eptflag - endpoint truncation flag (only has effect when imptyp = 0):
%                  0 (or not specified)  -  move data outside range to
%                                   nearest endpoint
%                  1  -  truncate data outside range
% Output:
%     hsjpi  -  Sheather Jones Plug In bandwidth choice
%
% Assumes path can find personal functions:
%    vec2matSM.m
%    lbinrSM.m
%    bwrfphSM.m
%    bwosSM.m
%    rootfSM

%    Copyright (c) J. S. Marron 1996-2011


%  Set parameters and defaults according to number of input arguments
%
if nargin == 1 ;    %  only 1 argument input, use default xgrid params
  ixgridp = 0 ;
else ;              %  xgrid was specified, use that
  ixgridp = xgridp ;
end ;

if nargin <= 2 ;    %  at most 2 arguments input, use default hgrid par's
  ihgridp = 0 ;
else ;              %  xgrid was specified, use that
  ihgridp = hgridp ;
end ;

if nargin <= 3 ;    %  at most 3 arguments input, use default implementation
  iimptyp = 0 ;
else ;              %  implementation was specified, use that
  iimptyp = imptyp ;
end ;

if nargin <= 4 ;    %  at most 4 arguments input, use default endpt trunc
  ieptflag = 0 ;    %  Default
else ;
  ieptflag = eptflag ;    %  Endpt trunc was specified, so use it
end ;


hsjpi = [] ;
if size(data,2) > 1 ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from bwsjpiSM.m:         !!!') ;
    disp('!!!   data must be a column vector   !!!') ;
    disp('!!!   Terminating Execution          !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    return ;
end ;



%  Bin, if needed
%
if iimptyp == -1 ;   %  Then data have already been binned

  if (length(ixgridp) == 1) | (length(ixgridp) > 3) ;
                       %  Then can't proceed because don't have bin ends
    disp('!!!   Error: bwsjpib needs to know the endpoints   !!!') ;
    disp('!!!            to use this implementation        !!!') ;
  else ;
    bincts = data ;

    nbin = 401 ;
    lend = ixgridp(1) ;
    rend = ixgridp(2) ;
    if length(ixgridp) == 3 ;          %  then use number of grid points
      nbin = ixgridp(3) ;
    end ;

    if nbin ~= length(bincts) ;    %  Then something is wrong
      disp('!!!   Warning: bwsjpib was told the wrong number of bins   !!!') ;
      disp('!!!            will just use the number of counts.       !!!') ;
      nbin = length(bincts) ;
    end ;
  end ;

  %  Get standard deviation from bin counts
  n = sum(bincts) ;
  bcents = linspace(lend,rend,nbin)' ;
  avg = (bincts' * bcents) / n ;
  sd = bcents - avg ;
  sd = sd .* sd ;    
          %  (x_i - avg)^2
  sd = bincts' * sd ;  
          %  sum((x_i - avg)^2)
  sd = sqrt(sd / (n - 1)) ;

else ;               %  Then need to bin data

  if length(ixgridp) > 3 ;  %  Then need to warn of change to default
    disp('!!!   Warning: gpkde was given an xgrid, and also   !!!') ;
    disp('!!!       asked to bin; will bin and ignore xgrid   !!!') ;
  end ;

  %  Specify grid parameters
  nbin = 401 ;         %  Default
  lend = min(data) ;   %  Default
  rend = max(data) ;   %  Default
  if (length(ixgridp) == 2) | (length(ixgridp) == 3) ;
                                   %  then use input endpoints
    lend = ixgridp(1) ;
    rend = ixgridp(2) ;
  end ;
  if length(ixgridp) == 3 ;          %  then use number of grid points
    nbin = ixgridp(3) ;
  end ;

  if lend > rend ;    %  Then bad range has been input
    disp('!!!   Error in gpkde: invalid range chosen  !!!') ;
    bincts = [] ;
  else ;
    bincts = lbinrSM(data,[lend,rend,nbin],ieptflag) ;
  end ;

  sd = std(data) ;

end ;
n = sum(bincts) ;
          %  put this here in case of truncations during binning



%  Get hgrid for rootfinding over
%
if length(ihgridp) < 2 ;
  if iimptyp == -1 ;   %  Then data have already been binned
    hmax = bwosSM([n,sd]) ;
  else ;
    hmax = bwosSM(data) ;
  end ;
  hmin = hmax / 9 ;
else ;
  hmin = ihgridp(1) ;
  hmax = ihgridp(2) ;
end ;

if length(ihgridp) < 3 ;
  nh = 21 ;
else ;
  nh = ihgridp(3) ;
end ;

vlogh = linspace(log10(hmin),log10(hmax),nh)' ;
vh = 10.^vlogh ;



%  Evaluate SJPI score function
%
rk = 1 / (2 * sqrt(pi)) ;
          %  Roughness of the Gaussian Kernel
mk2 = 1 ;
          %  Second moment of the Gaussian Kernel
mk22 = 1 ;
          %  Square of the second moment of the Gaussian Kernel
k40 = 3 / sqrt(2 * pi) ;
          %  4th deriv of Gaussian kernel, evaluated at 0

%  get pilot rfpp
rfpppr = sd^(-7) * 15 / (sqrt(pi) * 2^4) ;
rfpph = (2 * 3 / (sqrt(2 * pi) * rfpppr * n))^(1/7) ;
          %  got this from old GAUSS proc nmi2hh.g
rfpp = bwrfphSM(bincts,[lend,rend],rfpph,2) ;

%  get pilot rfppp
rfppppr = sd^(-9) * 105 / (sqrt(pi) * 2^5) ;
rfppph = (2 * 15 / (sqrt(2 * pi) * rfppppr * n))^(1/9) ;
          %  got this from old GAUSS proc nmi2hh.g
rfppp = bwrfphSM(bincts,[lend,rend],rfppph,3) ;

c = ((2 * k40) / rfppp)^(1/7) ;
c = c * (rfpp / rk)^(1/7) ;
c = c * mk2^(1/7) ;
va = c * vh.^(5/7) ;

vsjpisf = [] ;
for ih = 1:nh ;
  h = vh(ih) ;
  a = va(ih) ;

  rfppa = bwrfphSM(bincts,[lend,rend],a,2) ;
          %  Estimated roughness of f'', using bandwidth a 

  sjpisf = h - (rk / (rfppa * mk22 * n))^(1/5) ;

  vsjpisf = [vsjpisf; sjpisf] ;
end ;



%  Find root for final answer
%
hsjpi = rootfSM(vlogh,vsjpisf,+1,0) ;
          %  Use h's on log scale
          %  +1 to find upcrossings
          %  0 for largest root (upcrossing)
hsjpi = 10^hsjpi ;

