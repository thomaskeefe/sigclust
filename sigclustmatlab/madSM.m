function mad = madSM(data,iscale) 
% MADSM, Median Absolute Deviation (from the median)
%     Gives a robust estimate of scale for a univariate set of data
%   Can use 1 or 2 arguments.
%   Steve Marron's matlab function
% Inputs:
%     data - vector or matrix of data, when this is a matrix, the
%                 MAD of ech column is computed
%   iscale - scale of the output:
%               0 - on the raw scale of MAD
%               1 - (or unspecified) on the scale of the standard
%                         deviation of a Normal distribution
% Output:
%        mad - scalar MAD for a vector input,
%                   or row vector of MAD's for a matrix input
%
% Assumes path can find personal function:
%    vec2matSM.m

%    Copyright (c) J. S. Marron 1998, 2001


%  Set parameters according to number of input arguments
%
if nargin == 1 ;       %  only 1 argument input, use default iscale = 1 
  iiscale = 1 ;
else ;                 %  then use input value
  iiscale = iscale ;
end ;



%  calculate MAD
%
med = median(data) ;
if min(size(data)) > 1 ;   %  then have a matrix, so expand med
  med = vec2matSM(med,size(data,1)) ;
end ;
dev = data - med ;
          %  Deviation from the median
mad = median(abs(dev)) ;
          %  Median Absolute Deviation



if iiscale == 1 ;    %  then need to adjust to scale of 
                     %               normal distribution
  mad = mad / (norminv(.75) - norminv(.5)) ;
          %  adjust to scale of standard deviation
end ;




