function [bindat,bincent] = lbinrSM(data,vgridp,eptflag,ibtype) 
% LBINRSM, Linear BINneR (density and regression est.)
%   Steve Marron's matlab function
%     Does linear binning of either density or regression 1-d data,
%     to an equally spaced grid.
%   Can use first 1, 2, 3 or 4 arguments.
% Inputs:
%     data   - either n x 1 column vector of density estimation data
%                  or n x 2 matrix of regression data:
%                            X's in first column,  Y's in second
%     vgridp - vector of grid parameters:
%                  0 (or not specified)  -  use endpts of data and 401 bins
%                  [le; lr]  -  le is left end, re is right, 401 bins
%                         (get error message and no return if le > lr)
%                  [le; lr; nb] - le left, re right, and nb bins
%     eptflag - endpoint truncation flag:
%                  0 (or not specified)  -  move data outside range to
%                                   nearest endpoint
%                  1  -  truncate data outside range
%    ibtype - flag indicating binning type:
%                  0 - Simple (histogram) binning
%                            Note:  for larger data sets, this was MUCH
%                                   faster than matlab's HIST, 
%                                   for matlab versions before ~6.0,
%                                   but currently seems much slower
%                  1 - (or unspecified) - Linear binning 
%                            (default, when ibtype not specified)
% Output:
%     bindat  - binned data:
%                  nb x 1 column vector of bin counts for density estimation
%                  nb x 2 matrix, with (1) bin counts, (2) bin sum of Y's,
%                             for regression  {(2) / (1) gives bin avgs}
%     bincent - nbin x 1 vector of 
%                  bin centers, can also get this from linspace(le,re,nb)'  
%
% Used by:   kdeSM.m
%            nprSM.m
%

%    Copyright (c) J. S. Marron 1996-2011


%  Detect if this is density estimation, or regression, and handle data
%
if size(data,2) == 1 ;    %  Then is density estimation
  xdat = data(:,1) ;
  idatyp = 1 ;
else ;    %  The assume regression ;
  xdat = data(:,1) ;
  ydat = data(:,2) ;
  idatyp = 2 ;
end ;
n = length(xdat) ;


%  Set parameters and defaults according to number of input arguments
%
if nargin == 1 ;    %  only 1 argument input
  lend = min(xdat) ;
  rend = max(xdat) ;
  nbin = 401 ;
else ;              %  Then some grid parameters have been input
  if length(vgridp) == 1 ;    %  then use default grid
    lend = min(xdat) ;
    rend = max(xdat) ;
    nbin = 401 ;
  elseif length(vgridp) == 2 ;   % use given endpoints, but default number
    lend = vgridp(1) ;
    rend = vgridp(2) ;
    nbin = 401 ;
  else ;
    lend = vgridp(1) ;
    rend = vgridp(2) ;
    nbin = vgridp(3) ;
  end ;
end ;

if nargin <= 2 ;    %  Then at most 2 inputs, so use default endpt trunc.
  ieptflag = 0 ;    %  Default
else ;
  ieptflag = eptflag ;    %  Have value, so use it
end ;

if nargin <= 3 ;    %  Then at most 3 inputs, so use default bintype
  iibtype = 1 ;     %  Default
else ;
  iibtype = ibtype ;  %  Have value, so use it
end ;


if lend < rend ;   %  Have good end points, so proceed with binning

  %  Initialize count and sum vectors to 0
  %
  bxdat = zeros(nbin,1) ;
  if idatyp > 1 ;    %  Regression estimation
    bydat = zeros(nbin,1) ;
  end ;


  %  Work with data below bin range
  %
  loflag = ((xdat - lend) < (10^(-10) * (rend - lend))) ;
          %  this is a "numerically more robust" version of "xdat<=lend"
  numlo = sum(loflag) ;
  if numlo > 0 ;    %  If there are some below left end

    if numlo == n ;    %  Then all the data is below the left end, so:
      disp('!!! Caution from lbinrSM: all data below binning range !!!') ;
    end ;

    if ieptflag ~= 1 ;    %  Then move data to end, not truncate
      bxdat(1) = bxdat(1) + numlo ;
      if idatyp > 1 ;    %  Regression estimation
        bydat(1) = bydat(1) + sum(ydat(loflag)) ;
      end ;
    else ;    %  Then truncate (except for point at end)
      eqlendflag =  (xdat == lend)  ;
      bxdat(1) = bxdat(1) + sum(eqlendflag) ;
      if idatyp > 1 ;    %  Regression estimation
        bydat(1) = bydat(1) + sum(ydat(eqlendflag)) ;
      end ;
    end ;

  end ;


  %  Work with data above bin range
  %
  hiflag = ((xdat - rend) > -(10^(-10) * (rend - lend))) ;
          %  this is a "numerically more robust" version of "xdat >= rend"
  numhi = sum(hiflag) ;
  if numhi > 0 ;    %  If there are some above right end

    if numhi == n ;    %  Then all the data is above the right end, so:
      disp('!!! Caution from lbinrSM: all data above binning range !!!') ;
    end ;

    if ieptflag ~= 1 ;    %  Then move data to end, not truncate
      bxdat(nbin) = bxdat(nbin) + numhi ;
      if idatyp > 1 ;    %  Regression estimation
        bydat(nbin) = bydat(nbin) + sum(ydat(hiflag)) ;
      end ;
    else ;    %  Then truncate (except for point at end)
      eqrendflag =  (xdat == rend)  ;
      bxdat(nbin) = bxdat(nbin) + sum(eqrendflag) ;
      if idatyp > 1 ;    %  Regression estimation
        bydat(nbin) = bydat(nbin) + sum(ydat(eqrendflag)) ;
      end ;
    end ;

  end ;


  %  Work with interior data
  %
  iflag = (~loflag) & (~hiflag) ;        
  numi = sum(iflag) ;
  if numi > 0 ;    %  If there are some interior points

    ixdat = xdat(iflag) ;    %  Interior points
    if idatyp > 1 ;    %  Regression estimation
      iydat = ydat(iflag) ;    %  Interior points
    end ;

    isixdat = ((nbin - 1) * (ixdat - lend) ./ (rend - lend)) + 1 ;
          %  linear transformation, that maps  lend ---> 1
          %                              and   rend ---> nbin

    if iibtype == 0 ;    % Then do simple (histogram) binning

      vibinc = floor(isixdat + .5) ;
          %  indices of closest bin centers
      for idati = 1:numi ;    %  loop through data points in interior
        bxdat(vibinc(idati)) = bxdat(vibinc(idati)) + 1 ;
          %  put one in bin, for each data point
      end ;
          %  Implementation note:  Loops such as this seem to be needed,
          %  since the obvious matrix version:
          %       bxdat(vibinc) = bxdat(vibinc) + ones(numi,1)
          %  would not m updates, when there m duplications in vibinc,
          %  but instead would only do one (the last one).
      if idatyp > 1 ;    %  Regression estimation
        for idati = 1:numi ;    %  loop through data points in interior
          bydat(vibinc(idati)) = bydat(vibinc(idati)) + iydat(idati) ;
          %  add each y to appropriate bin
        end ;
      end ;

    else ;    %  Then do linear binning (default)

      vibinl = floor(isixdat) ;
          %  indices of bin center to left (integer part of isixdat)
      vwt = isixdat - vibinl ;
          %  weights to use in linear binning (fractional part)

      for idati = 1:numi ;    %  loop through data points in interior
        bxdat(vibinl(idati)) = bxdat(vibinl(idati)) + (1 - vwt(idati)) ;
          %  update of bins on left side
          %  put (1 - wt) in bin, for each data point
        bxdat(vibinl(idati) + 1) = bxdat(vibinl(idati) + 1) + vwt(idati) ;
          %  update of bins on right side
          %  put wt in bin, for each data point
      end ;
      if idatyp > 1 ;    %  Regression estimation
        for idati = 1:numi ;    %  loop through data points in interior
          bydat(vibinl(idati)) = bydat(vibinl(idati)) + ...
                                        (1 - vwt(idati)) .* iydat(idati) ;
          %  update of bins on left side
          %  add each (1 - wt) * y to appropriate bin
          bydat(vibinl(idati) + 1) = bydat(vibinl(idati) + 1) + ...
                                           vwt(idati) .* iydat(idati) ;
          %  update of bins on right side
          %  add each wt * y to appropriate bin
        end ;
      end ;

    end ;

  end ;


  %  Combine results, and output
  %
  bindat = bxdat ;
  if idatyp > 1 ;    %  Regression estimation
    bindat = [bindat bydat] ;
  end ;
  bincent = linspace(lend,rend,nbin)' ;



else ;    %  Then give error message since range is invalid
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error in lbinrSM: invalid binning range   !!!') ;
  disp(['!!!   Need (lrnd = ' num2str(lend) ') < (rend = ' num2str(lend) ')']) ;
  if  (lend == rend)  &  (idatyp == 1)  ;
    disp('!!!   Can be caused by all inputs the same      !!!') ;
  elseif  (lend == rend)  &  (idatyp == 2)  ;
    disp('!!!   Can be caused by all x inputs the same    !!!') ;
  end ;
  disp('!!!   Giving Empty Returns                      !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  bindat = [] ;
  bincent = [] ;
end ;

