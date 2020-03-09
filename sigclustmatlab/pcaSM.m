function outstruct = pcaSM(mdat,paramstruct) 
% PCASM, Principal Component Analysis
%     Computes eigenvalues and eigenvectors for data vectors
%     in mdat.  Eigenvectors are orthogonal directions of
%     greatest variability, and eigenvalues are sums of
%     squares of projections of data in those directions
%   Steve Marron's matlab function
%     
% Inputs:
%
%   mdat        - d x n matrix of multivariate data
%                         (each col is a data vector)
%                         d = dimension of each data vector
%                         n = number of data vectors 
%
%   paramstruct - a Matlab structure of input parameters
%                    Use: "help struct" and "help datatypes" to
%                         learn about these.
%                    Create one, using commands of the form:
%
%       paramstruct = struct('field1',values1,...
%                            'field2',values2,...
%                            'field3',values3) ;
%
%                          where any of the following can be used,
%                          these are optional, misspecified values
%                          revert to defaults
%
%    fields            values
%
%    npc              number of principal directions to compute
%                     0  (default)  compute "full set" = rank(mdat)
%                            this cannot be bigger than rank of mdat
%                            when doing dual eigen decomposition
%                                (otherwise will reset to rank(mdat))
%
%    idual            1  force use of dual eigen decomposition
%                     0  (default)  choose smallest
%                     -1  force use of direct eigen decomposition
%
%    iorient          0  use orientation (+ or -) that comes from Matlab
%                             eigen decomposition
%                     1  (default)  re-orient each eigenvector (i.e. multiply
%                             by +-1), to make max projection go in positive 
%                             direction.
%                     2  re-orient each eigenvector to make largest 2nd moment
%                             (on each side of the origin) go in positive direction
%
%    iprestd          0  (default)  treat as raw data and standardize,
%                             i.e. subtract mean vector, 
%                                     and mult. by 1/sqrt(n-1)
%                     1  treat as pre-standardized data, i.e.
%                             leave mean and scale alone
%                             (note assumes 1/sqrt(n-1) 
%                                 normalization has been done)
%
%    iscreenwrite     0  (default)  no screen writes
%                     1  write to screen to show progress
%
%    viout            vector of indices 0 or 1 for various types of output
%                                   1 in a given entry will include that 
%                                   in output structure
%                     Entry 1:  veigval  -  npc x 1 vector of 1st npc  
%                                   eigenvalues, sorted in decreasing order
%                                   Need to multiply by (n-1) for these to
%                                   work like sums of squares
%                     Entry 2:  meigvec  -  d x npc matrix of corresponding
%                                   eigenvectors
%                                   These are orthonormal
%                                   Entries are typically called "loadings"
%                     Entry 3:  vmean  -  d x 1 mean vector
%                                   (returns assumed value of zeros(1,n)
%                                    for iprestd = 1) 
%                     Entry 4:  mmeanresid  -  d x n matrix of 
%                                   residuals from mean
%                                   (returns assumed value of
%                                    mdat * sqrt(n-1), for iprestd = 1) 
%                     Entry 5:  mpc  -  npc x n matrix of 
%                                   "principal components"
%                                   i.e. of coefficients of projections 
%                                   of data onto eigenvectors, 
%                                   i.e. of "scores".
%                                   Note can "recover data" from
%                                     vmean + meigvec * mpc
%                                   Common "PCi" projection plot comes 
%                                   from meigvec(:,i) * mpc(i,:)
%                     Entry 6:  a3proj  -  d x n x npc  3d array of
%                                   projections of data onto individual
%                                   eigenvectors.
%                                   Use this for PCi projection plots
%                     Entry 7:  sstot  -  1 x 1  scalar, with
%                                   overall Sum of Squares of mdat
%                     Entry 8:  ssmresid  -  1 x 1  scalar, with
%                                   mean residual Sum of Squares
%                                   (divide by sstot, and multiply
%                                       by 100, to get percent of sstot)
%                     Entry 9:  vpropSSmr  -  npc x 1  vector of 
%                                   proportions of SS, 
%                                   relative to meanresid SS
%                                   (multiply by 100 to get percent)
%                                   (use cumsum, for "cumulative power")
%                     Entry 10:  vpropSSpr  -  npc x 1  vector of 
%                                   proportions of SS, 
%                                   relative to Previous Residual SS
%                                   (relative to mean resid, for 1st)
%                                   (multiply by 100 to get percent)
%                         When viout has length < 10, will pad with 0's
%                         Default is [1, 1]
%
% Output:
%
%    outstruct  - structure with outputs, as requested in viout above
%
%        Note: unpack these with commands of the form:
%                  veigval = getfield(outstruct,'veigval') ;
%
%          The covariance matrix of the data admits the decomposition:
%               covariance = meigvec * diag(veigval) * meigvec'
%               i.e. meigvec' * covariance * meigvec = diag(veigval)
%
% Assumes path can find personal function:
%    vec2matSM.m

%    Copyright (c) J. S. Marron 2001, 2004



%  First set all parameter to defaults
%
npc = 0 ;
inpcset = 0 ;
idual = 0 ;
iorient = 1 ;
iprestd = 0 ;
iscreenwrite = 0 ;
viout = [1, 1] ;



%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 1 ;   %  then paramstruct is an argument

  if isfield(paramstruct,'npc') ;    %  then change to input value
    npc = getfield(paramstruct,'npc') ; 
          %  record fact that npc was set 
  end ;

  if isfield(paramstruct,'idual') ;    %  then change to input value
    idual = getfield(paramstruct,'idual') ; 
  end ;

  if isfield(paramstruct,'iorient') ;    %  then change to input value
    iorient = getfield(paramstruct,'iorient') ; 
  end ;

  if isfield(paramstruct,'iprestd') ;    %  then change to input value
    iprestd = getfield(paramstruct,'iprestd') ; 
  end ;

  if isfield(paramstruct,'iscreenwrite') ;    %  then change to input value
    iscreenwrite = getfield(paramstruct,'iscreenwrite') ; 
  end ;

  if isfield(paramstruct,'viout') ;    %  then change to input value
    viout = getfield(paramstruct,'viout') ; 
  end ;


end ;  %  of resetting of input parameters



%  Readjust viout as needed
%
maxviout = 10 ;
    %  largest useful size for viout 
if size(viout,1) > 1 ;    %  if have more than one row
  viout = viout' ;
end ;
if size(viout,1) == 1 ;    %  then have row vector

  if length(viout) < maxviout ;    %  then pad with 0s
    viout = [viout zeros(1,maxviout - length(viout))] ;
  end ;

else ;    %  invalid viout, so indicate and quit

  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from pcaSM:       !!!') ;
  disp('!!!   Invalid viout           !!!') ;
  disp('!!!   Terminating exceution   !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;

end ;



%  get dimensions from data matrix
%
d = size(mdat,1) ;
          %  dimension of each data vector (col)
n = size(mdat,2) ;
          %  number of data vectors (cols)



%  Calculate total sum of squares
%
if viout(7) == 1 ;

  sstot = sum(sum(mdat.^2)) ;

end ;



if iprestd ~= 1 ;    %  then need to standardize input data matrix

  vmean = mean(mdat,2) ;
          %  mean across rows

  mdat = mdat - vec2matSM(vmean,n) ;
          %  recenter data point cloud

  if  viout(4) == 1  | ...
      viout(8) == 1  | ...
      viout(9) == 1  | ...
      viout(10) == 1  ;    %  need to output mean residuals
    mmeanresid = mdat ;
  end ;

  mdat = (1 / sqrt(n - 1)) * mdat ;
          %  puts on "covariance scale", so that covariance matrix
          %  is just "outer product":  mdat * mdat'


else ;

  if viout(3) == 1 ;    %  need to output actual mean (despite assumption)
    vmean = zeros(d,1) ;
          %  assumed mean vector
  end ;

  if  viout(4) == 1  | ...
      viout(8) == 1  | ...
      viout(9) == 1  | ...
      viout(10) == 1  ;    %  need to output actual mmeanresid
                           %                (despite assumption)
    mmeanresid = mdat * sqrt(n - 1) ;
          %  assumed mean residuals
  end ;

  if iscreenwrite == 1 ;
    disp('!!!    pcaSM assuming that data are standardized   !!!') ;
  end ;


end ;

rankmdat = rank(mdat) ;

if npc == 0 ;    %  Then compute all non-zero eigenvalues
  npc = rankmdat ;
end ;

if npc > rankmdat ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!  Warning from pcaSM:  npc too large  !!!') ;
  disp('!!!  resetting to rank of mdat           !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  npc = rankmdat ;
end ;



if  (idual ~= 1)  &  (idual ~= -1)  ;    %  then need to use inputs
                                         %  to choose idual
  if d > n ;
    idual = 1 ;
          %  dual computation is simplest
  else ;
    idual = -1 ;
          %  direct computation is simplest
  end ;
  
end ;




%  set additional parameters
%
eigsopts = struct('disp', 0) ;
          %  Option structure for eigs, this supresses output



if iscreenwrite == 1 ;
  disp('    pcaSM starting eigendecomposition') ;
end ;



if idual == -1 ;    %  do direct eigen decomposition

  sigmahat = mdat * mdat' ;
          %  direct covariance is outer product of normalized data matrix

  if npc < d ;    %  do reduced eigen decomposition
  
    [B,D] = eigs(sigmahat,npc,'LA',eigsopts) ;
          %  partial eigenvalue decomp., using only npc largest
          %  'LR' for Largest Real part

    veigval = diag(D) ;
    meigvec = B ;

  else ;    %  do full eigendecomposition

    [B,D] = eig(sigmahat) ;

    veigval = diag(D) ;
    meigvec = B ;

    %  careful need to do a sort
    %
    [veigval,sortind] = sort(veigval) ;
        %  sorted in increasing order
    veigval = flipud(veigval) ;
    sortind = flipud(sortind) ;
        %  sorted in decreasing order
    meigvec = meigvec(:,sortind) ;
        %  put in same order

  end ;


elseif idual == 1 ;    %  do dual eigen decomposition

  sigmastar = mdat' * mdat ;
          %  dual covariance is inner product of normalized data matrix

  [Bstar,Dstar] = eigs(sigmastar,npc,'LA',eigsopts) ;
          %  partial eigenvalue decomp., using only npc largest
          %  'LR' for Largest Real part


  veigval = diag(Dstar) ;
  irDstar = diag(1 ./ sqrt(veigval)) ;
  meigvec = mdat * Bstar * irDstar ;


end ;



if  viout(5) == 1  | ...
    iorient == 1  | ...
    iorient == 2  ;    %  then compute projections of data on eigenvectors

  mpc = sqrt(n - 1) * meigvec' * mdat ;
      %  npc x n matrix of projections of data on eigenvectors


  if iorient == 1 ;    %  then re-orient direction +- of eigenvectors
                       %  to make max projection go in positive direction

    if iscreenwrite == 1 ;
      disp('    pcaSM re-orienting eigenvectors') ;
    end ;

    for ipc = 1:npc ;    %  loop through eigenvectors

      vpc = mpc(ipc,:)' ;
          %  vector of projections on this eigenvector
      [temp,idat] = max(abs(vpc)) ;
          %  index of largest magnitude projection
      flipflag = sign(vpc(idat)) ;
          %  = or - sign of largest magnitude projection

      if flipflag < 0 ;    %  then flip this eigenvector and the projections
        meigvec(:,ipc) = -meigvec(:,ipc) ;
        mpc(ipc,:) = -mpc(ipc,:) ;
      end ;

    end ;


  elseif iorient == 2 ;    %  then re-orient direction +- of eigenvectors
                           %  to make largest 2nd moment (on each side of 
                           %  the origin) go in positive direction

    if iscreenwrite == 1 ;
      disp('    pcaSM re-orienting eigenvectors') ;
    end ;

    for ipc = 1:npc ;    %  loop through eigenvectors

      vpc = mpc(ipc,:)' ;
          %  vector of projections on this eigenvector
      flagp = vpc > 0 ;
          %  one where there are positive projections
      np = sum(flagp) ;
          %  number of positive projections
      flagn = vpc < 0 ;
          %  one where there are negative projections
      nn = sum(flagn) ;
          %  number of positive projections
      if  nn > 0  &  np > 0  ;
        smp = sum(vpc(flagp) .^2) / np ;
        smn = sum(vpc(flagn) .^2) / nn ;
        if smp < smn ;
          flipflag = -1 ;
        else ;
          flipflag = 1 ;
        end ;
      elseif  nn == 0  &  np > 0  ;
        flipflag = 1 ;
      elseif  nn > 0  &  np == 0  ;
        flipflag = -1 ;
      else ;
        flipflag = 0 ;
      end ;  

      if flipflag < 0 ;    %  then flip this eigenvector and the projections
        meigvec(:,ipc) = -meigvec(:,ipc) ;
        mpc(ipc,:) = -mpc(ipc,:) ;
      end ;

    end ;


  end ;


end ;






if viout(6) == 1 ;

  %  create 3 dim array versions
  %
  a3meigvec = [] ;
  for i = 1:n ;
    a3meigvec = cat(3,a3meigvec,meigvec) ;
  end ;
  a3meigvec = permute(a3meigvec,[1 3 2]) ;
      %  essentially "transpose" in 3d
      %  need to swap 2nd and 3rd dimensions 
      %  to have give  d x n x npc  result

  a3mpc = [] ;
  for i = 1:d ;
    a3mpc = cat(3,a3mpc,mpc) ;
  end ;
  a3mpc = permute(a3mpc,[3 2 1]) ;
      %  essentially "transpose" in 3d
      %  need to swap 1st and 3rd dimensions 
      %  to have give  d x n x npc  result

  a3proj = times(a3meigvec,a3mpc) ;
      %  high d version of usual .* operation, gives:
      %    d x n x npc  3d array of projections of data 
      %                   onto individual eigenvectors.

end ;



%  Calculate needed sum of squares
%
if  viout(8) == 1  | ...
    viout(9) == 1  | ...
    viout(10) == 1  ;

  ssmresid = sum(sum(mmeanresid.^2)) ;

end ;

if  viout(9) == 1  | ...
    viout(10) == 1  ;

  vpcSS = (n - 1) * veigval ;
      %  sum of squares of pc's in each eigen-direction
  vpropSSmr = vpcSS / ssmresid ;

end ;

if viout(10) == 1 ;

  prevresidSS = ssmresid ;
      %  previous residual Sum of Squares
  residSS = ssmresid - vpcSS(1) ;
      %  current residual Sum of Squares
  vpropSSpr = residSS / prevresidSS ;

  for iev = 2:npc  ;
    prevresidSS = residSS ;
    residSS = prevresidSS - vpcSS(iev) ;
    if residSS < (10^(-12) * ssmresid) ;    %  then any difference with 0
                                            %  is regarded as round off errors
                                            %  so set to 0, for output
      residSS = 0 ;
    end ;
    vpropSSpr = [vpropSSpr; (residSS / prevresidSS)] ;

  end ;

end ;



if iscreenwrite == 1 ;
  disp('    pcaSM finished with pca calculation') ;
  disp('  ') ;
end ;



%  Construct output structure
%
outstruct = struct('dummy',[]) ;
    %  put in dummy field, to avoid empty structure

if viout(1) == 1 ;
  outstruct = setfield(outstruct,'veigval',veigval) ;
end ;

if viout(2) == 1 ;
  outstruct = setfield(outstruct,'meigvec',meigvec) ;
end ;

if viout(3) == 1 ;
  outstruct = setfield(outstruct,'vmean',vmean) ;
end ;

if viout(4) == 1 ;
  outstruct  = setfield(outstruct,'mmeanresid',mmeanresid) ;
end ;

if viout(5) == 1 ;
  outstruct  = setfield(outstruct,'mpc',mpc) ;
end ;

if viout(6) == 1 ;
  outstruct  = setfield(outstruct,'a3proj',a3proj) ;
end ;

if viout(7) == 1 ;
  outstruct  = setfield(outstruct,'sstot',sstot) ;
end ;

if viout(8) == 1 ;
  outstruct  = setfield(outstruct,'ssmresid',ssmresid) ;
end ;

if viout(9) == 1 ;
  outstruct  = setfield(outstruct,'vpropSSmr',vpropSSmr) ;
end ;

if viout(10) == 1 ;
  outstruct  = setfield(outstruct,'vpropSSpr',vpropSSpr) ;
end ;

outstruct = rmfield(outstruct,'dummy') ;





