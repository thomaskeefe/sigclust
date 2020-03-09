function  [veigvest, tau] = SigClustCovEstSM(vsampeigv,sig2b) 
% SIGCLUSTCOVESTSM, statistical SIGnificance of CLUSTers,
%         computes soft thresholded COVariance ESTimate, 
%         with the choice of tau that best matches signal 
%         power, for given set of eigenvalues  
%         and given background variance
%   This is the soft thresholded version published in JCGS, 2014
%   Steve Marron & Hanwen Huang's matlab function
%     Provides fundamental tool for studying the
%     consistency of the standard 2-means clustering
%     method for splitting a given data set.
%
% Inputs:
%
%    vsampeigv  - d x 1 vector of sample eigenvalues
%                    must be in decreaszing order & >= 0
%
%    sig2b      - scalar, given background noise level, 
%                  on the scale of variance 
%
% Outputs:
%
%    veigvest         d x 1 vector of estimated eigenvalues,
%                     soft thresholded to sig2b,
%                     to match signal power
%
%    tau              Soft Threshold Value that gives best
%                     match of signal power 
%
%
% Assumes path can find personal functions:


%    vec2matSM.m

%    Copyright (c) J. S. Marron & Hanwen Huang 2008, 2014




%  set preliminary stuff
%
d = size(vsampeigv,1) ;
         %  dimension of data & covariance matrix
if size(vsampeigv,2) > 1 ;
  veigvest = [] ;
  tau = [] ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from SigClustCovEstSM:        !!!') ;
  disp('!!!   vsampeigv must be a column vector   !!!') ;
  disp('!!!   Returning Empty Solutions and       !!!') ;
  disp('!!!   Terminating Execution               !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;



%  Check that eigevalues have been sorted and are nonnegative
%
vdiff = vsampeigv(2:end) - vsampeigv(1:(end-1)) ;
flag = (vdiff > 0) ;
    %  one where have an increase in sample eigenvalues

if sum(flag) > 0 ;    %  then have an increase somewhere
  veigvest = [] ;
  tau = [] ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from SigClustCovEstSM:            !!!') ;
  disp('!!!   vsampeigv must be in decreasing order   !!!') ;
  disp('!!!   Returning Empty Solutions and           !!!') ;
  disp('!!!   Terminating Execution                   !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;
if min(vsampeigv) < 0 ;    %  then have a negative eigenvalue
  veigvest = [] ;
  tau = [] ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from SigClustCovEstSM:       !!!') ;
  disp('!!!   Need entries of vsampeigv >= 0     !!!') ;
  disp('!!!   Returning Empty Solutions and      !!!') ;
  disp('!!!   Terminating Execution              !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;



%  Check have some eigenvalues < sig2b
%
vtaucand = vsampeigv - sig2b ;
    %  vector of initial candidates for the threshold tau

if vtaucand(end) >= 0 ;    %  if all of these are positive
  veigvest = vsampeigv ;
      %  in this case, just use sample eigenvalues
  tau = 0 ;
      %  no adjustment by tau needed
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Warning from SigClustCovEstSM:           !!!') ;
  disp('!!!   smallest sample eigenvalue >=  sig2b,    !!!')
  disp('!!!   adjustment won''t change anything,        !!!') ;
  disp('!!!   just returning sample eigenvalues,       !!!') ;
  disp('!!!   Terminating Execution                    !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;



%  Check that sample eigenvalues have enough total power
%
totpow = sum(vsampeigv) ;
    %  total signal power in data = sum of eigenvalues

if totpow <= d * sig2b ;  
  veigvest = sig2b * ones(d,1) ;
      %  set output to constant value of sig2b
  tau = vtaucand(1) ;
      %  set this to smallest threshold,
      %  which will make bring all values down to sig2b
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Warning from SigClustCovEstSM:           !!!') ;
  disp('!!!   Not enough total signal power,           !!!') ;
  disp('!!!       for this choice of sig2b.            !!!') ;
  disp('!!!   Returning Flat Covariance Est.           !!!') ;
  disp('!!!       which may have more total power,     !!!') ;
  disp('!!!       making SigClust anti-convervative    !!!') ;
  disp('!!!   Recommendation: take careful look at     !!!') ;
  disp('!!!       SigClust Diagnostic plots            !!!') ;
  disp('!!!   Terminating Execution                    !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;



%  Find threshold, to preserve total signal power
%
[temp, icut] = max(vtaucand <= 0) ;
    %  index of first element of vtaucan that is <= 0
icut = icut - 1 ;
    %  index of last eigenvalue, with taucand > 0 

powertail = sum(vsampeigv((icut + 1):end)) ;
    %  total power for eignevalues beyond icut

power2shift = sig2b * (d - icut) - powertail ;
    %  total of power to be shifted into tail, by thresholding

vi = (1:icut)' ;
vcumtaucand = flipud(cumsum(flipud(vtaucand(vi)))) ;
    %  vector of the tail cumulatives of the sample eigenvalues,
    %        starting with eigenvalues above the cutoff
    %  i.e. the sum over this eigenvalue 
    %        and all smaller ones, down to icut

vpowershifted = (vi - 1) .* vtaucand(1:icut) + vcumtaucand ;
    %  vector of power shifted by thresholding,
    %  for each candidate tau

flag = vpowershifted < power2shift ;
    %  one where not enough power shifted
if sum(flag) == 0 ;    %  then itau should include everything
  itau = 0 ;
      %  use this to flag case where everything is included
      %  in this case use icut as index
else ;    %  then find itau using max
  [temp,itau] = max(flag) ;
      %  index of first element with not enough power shifted
end ;

if itau == 1 ;
  powerprop = power2shift / vpowershifted ;
    %  proportion of desired power, relative to candidate powers
  tau = powerprop * vtaucand(1) ;
    %  interpolated value of tau, 
    %  that moves power in the amount of power2shift
elseif itau == 0 ;    %  then choose tau based only on last
  powerprop = power2shift / vpowershifted(icut) ;
    %  proportion of desired power, relative to candidate powers
  tau =  powerprop * vtaucand(icut) ;
    %  interpolated value of tau, 
    %  that moves power in the amount of power2shift
else ;    %  need to do linear interpolation
  powerprop = (power2shift - vpowershifted(itau)) / ...
                  (vpowershifted(itau - 1) - vpowershifted(itau)) ;
    %  proportion of desired power, relative to candidate powers
  tau = vtaucand(itau) + powerprop * (vtaucand(itau - 1) - vtaucand(itau)) ;
    %  interpolated value of tau, 
    %  that moves power in the amount of power2shift
end ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add adjustment for nHanwen Huang's 2014 JCGS
% soft threshold
% These lines are only change
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
taub = 0;
tauu = tau;
etau = (tauu-taub)/100;
ids = zeros(100,1);
for i = 1:100
    taus = taub + (i-1)*etau;
    eigvaltemp = vsampeigv - taus;
    eigvaltemp(eigvaltemp<sig2b) = sig2b;
    ids(i) = eigvaltemp(1)/sum(eigvaltemp);
end 
[vlmax,idmax] = max(ids);
tau = taub + (idmax-1)*etau;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Construct final output
%
veigvest = vsampeigv - tau ;
    %  thresholded sample eigenvalues
flag = veigvest > sig2b ;
    %  one where thresholded eigenvalues big enough to keep
veigvest = flag .* veigvest + (1 - flag) .* (sig2b * ones(d,1)) ;
    %  replaced thresholded eigenvalues smaller than sig2b by sig2b



%  Used these lines for checking:
%
%disp(['    Total Signal Power = ' num2str(totpow)]) ;
%finsigpow = sum(veigvest) ;
%disp(['    Final Signal Power = ' num2str(finsigpow)]) ;
%disp(['    Absolute Difference = ' num2str(abs(finsigpow - totpow))]) ;




