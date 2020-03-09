function vroot = rootfSM(vx,vy,irtyp,lflag,iout) 
% ROOTFSM, ROOT Finder.
%   Steve Marron's matlab function
%     Approximates roots of a discretized continuous function,
%         fitting a cubic to nearest subintervals.
%   Can use first 2, 3, 4 or 5 arguments.
% Inputs:
%     vx    - column vector of arguments of a function to find root of.
%                   !!!  ASSUMED TO BE ORDERED  !!!
%     vy    - col. vec. of corresponding values of the function.
%     irtyp - root type to find:
%                downcrossing (i.e. sign change from - to +, local 
%                    min of integral), when irtyp < 0 (or unspecified)
%                any root, when irtyp = 0
%                upcrossing (i.e. sign change from + to -, local 
%                    max of integral), when irtyp > 0 
%     lflag - local root handling flag.  Finds:
%                smallest local root, when lflag < 0
%                largest local root, when lflag >= 0 (or unspecified) 
%     iout  - integer for type of output:
%                1 (or not included) - output only root x value
%                2      - output vec. with x and number of local roots
%                3      - output vec. with x, #roots and "error flag".
% Output:
%    vroot  - column vector with:
%               root x, when iout = 1 (or not included in call)
%               root x and # roots (i.e. sign changes), when iout = 2
%               root x, # roots, and "error flag" when iout = 3
%                     error flag has values:
%                            = -1, when no roots, but left is closer
%                            = 0,  when root is in interior
%                            = 1,  when no roots, but right is closer
%
%  See Also:  minrSM.m

%    Copyright (c) J. S. Marron 1996-2001


%  Set parameters and defaults according to number of input arguments
if nargin == 2 ;    %  only 2 arguments input
  iirtyp = -1 ;      
          %  Use default: root type is downcrossing
else ;              %  more than two arguments input
  iirtyp = irtyp ;
          %  Use input local root type
end ;

if nargin <= 3 ;    %  2 or 3 arguments input
  ilflag = 1 ;      
          %  Use default: largest local root
else ;              %  more than 4 arguments input
  ilflag = lflag ;
          %  Use input local root flag
end ;

if nargin <= 4 ;    %  2,3 or 4 arguments input
  iiout = 1 ;
          %  Use default: Output only root value
else ;              %  5 arguments input
  iiout = iout ;
          %  Use input choice of output type
end ;

nx = length(vx) ;


%  Find indices of important roots
yflag = (vy >= 0) ;
          %  flag sites where y is positive
yflagdif = yflag(2:nx) - yflag(1:(nx-1)) ;
          %  -1 at gaps with downcrossing
          %   0 at gaps with no sign change
          %  +1 at gaps with upcrossing
if iirtyp < 0 ;    %  If want to look at downcrossings 
  jumpflag = (yflagdif < 0) ;
          %  flag gaps where yflag decreases across 0
elseif iirtyp == 0 ;    %  If want to look at all 0 crossings 
  jumpflag = (abs(yflagdif) > 0) ;
          %  flag gaps where yflag crosses 0
else ;    %  If want to look at upcrossings 
  jumpflag = (yflagdif > 0) ;
          %  flag gaps where yflag increases across 0
end ;
njump = sum(jumpflag) ;
          %  number of interesting jumps,m i.e. zero-crossings


if njump == 0 ;    %  Then no interesting zero crossings

  nup = sum(yflag) ;    
          %  Number of sites where y positive

  if iirtyp < 0 ;        %  Then were looking for downcrossings
                         %  so take endpt "closest to being one"

    if nup == nx ;    %  then all sites are positive
                     %  so right end is "closest to downcrossing"
      xroot = vx(nx) ;
      errflag = 1 ;
    elseif nup == 0 ; %  then all sites are negative
                     %  so left end is "closest to downcrossing"
      xroot = vx(1) ;
      errflag = -1 ;
    else ;           %  then there was an upcrossing
                     %  so take endpoint "closest to 0"
      if abs(vy(1)) < abs(vy(nx)) ;
        xroot = vx(1) ;
        errflag = -1 ;
      else ;
        xroot = vx(nx) ;
        errflag = 1 ;
      end ;
    end ;

  elseif iirtyp > 0 ;    %  Then were looking for upcrossings
                         %  so take endpt "closest to being one"

    if nup == nx ;    %  then all sites are positive
                     %  so left end is "closest to downcrossing"
      xroot = vx(1) ;
      errflag = -1 ;
    elseif nup == 0 ; %  then all sites are negative
                     %  so right end is "closest to downcrossing"
      xroot = vx(nx) ;
      errflag = 1 ;
    else ;           %  then there was a downcrossing
                     %  so take endpoint "closest to 0"
      if abs(vy(1)) < abs(vy(nx)) ;
        xroot = vx(1) ;
        errflag = -1 ;
      else ;
        xroot = vx(nx) ;
        errflag = 1 ;
      end ;
    end ;

  elseif iirtyp == 0 ;    %  Then were looking for all crossings,
                         %  so take endpoint "closest to 0" 

    if abs(vy(1)) <= abs(vy(nx)) ;
      xroot = vx(1) ;
      errflag = -1 ;
    else ;
      xroot = vx(nx) ;
      errflag = 1 ;
    end ;

  end ;

else ;    %  Then work with zero crossings

  %  find index of interval where most interesting crossing occurs
  if ilflag < 0 ;    %  then want smallest local root
    [temp, i] = max(jumpflag) ;
  else ;             %  then want largest local root
   [temp, i] = max(flipud(jumpflag)) ;
    i = nx - i ;
  end ;

  if i == 1 ;               %  Then at left end, so fit parabola
    locx = vx(1:3) ;
    locy = vy(1:3) ;
          %  x & y values around current root
    pcoeffs = polyfit(locx,locy,2) ;
          %  Coefficients of interpolating quadratic
    rootvec = roots(pcoeffs) ;
          %  vector of roots
    xroot = rootvec((rootvec >= locx(1)) & (rootvec <=locx(2))) ;
          %  roots in desired range
    xroot = min(xroot) ;
          %  in case of multiple roots, take smallest
  elseif i == (nx - 1) ;    %  Then at right end, so fit parabola
    locx = vx(nx-2:nx) ;
    locy = vy(nx-2:nx) ;
          %  x & y values around current root
    pcoeffs = polyfit(locx,locy,2) ;
          %  Coefficients of interpolating quadratic
    rootvec = roots(pcoeffs) ;
          %  vector of roots
    xroot = rootvec((rootvec >= locx(2)) & (rootvec <=locx(3))) ;
          %  roots in desired range
    xroot = min(xroot) ;
          %  in case of multiple roots, take smallest
  else ;                    %  Then in interior, so fit cubic
    locx = vx((i-1):(i+2)) ;
    locy = vy((i-1):(i+2)) ;
          %  x & y values around current root
    pcoeffs = polyfit(locx,locy,3) ;
          %  Coefficients of interpolating cubic
    rootvec = roots(pcoeffs) ;
          %  vector of roots of cubic
    realflag = imag(rootvec) == 0 ;
          %  ones where entry is real (i.e. imaginary part is 0
    rootvec = rootvec(realflag) ;
          %  restrict to just real roots
    xroot = rootvec((rootvec >= locx(2)) & (rootvec <=locx(3))) ;
          %  roots in desired range
    xroot = min(xroot) ;
          %  in case of multiple roots, take smallest
  end ;
  errflag = 0 ;

end ;


if errflag == -1 ;
  disp('!!!   Warning: rootfSM hit left end   !!!') ;
elseif errflag == 1 ;
  disp('!!!   Warning: rootfSM hit right end   !!!') ;
end ;


%  Construct output vector
if iiout == 1 ;   %  then only output x root
  vroot = xroot ;
elseif iiout == 2 ;   %  then output x & # roots
  vroot = [xroot; njump] ;
elseif iiout == 3 ;   %  then output x, # roots, and errflag
  vroot = [xroot; njump; errflag] ;
end ;

