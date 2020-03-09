function ClustInd = ClustIndSM(mdata,Class1Flag,Class2Flag) 
% CLUSTINDSM, CLUSTerINDex
%     Index of clustering, that underlies 2-means (k = 2) clustering
%     using:  Squared Euclidean distance
%   Steve Marron's matlab function
% Inputs:
%     mdata      - d x n matrix of multivariate data
%                       (each col is a data vector)
%                       d = dimension of each data vector
%                       n = number of data vectors 
%     Class1Flag - 1 x n logical vector indicating elements of Class 1
%     Class2Flag - 1 x n logical vector indicating elements of Class 2
%                       There must be a one in a common element of these,
%                       But OK for both to have a zero
%                           (will just leave those out of calculation)
%                       Otherwise gives a warning message,
%                       and returns an empty value of Cluster Index
% Output:
%     ClustInd   - 2-means Cluster Index
%                       This is sum of within Class Sums of Squares about mean,
%                       divided by Total Sum of Squares about overall mean 
%
% Assumes path can find personal function:
%    vec2matSM.m

%    Copyright (c) J. S. Marron 2007


d = size(mdata,1) ;
n = size(mdata,2) ;

%  Check Inputs
%
if size(Class1Flag,1) ~= 1 ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:              !!!') ; 
  disp('!!!   Class1Flag needs to be a row vector   !!!') ;
  disp('!!!   Terminating Execution                 !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;

if size(Class2Flag,1) ~= 1 ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:              !!!') ; 
  disp('!!!   Class2Flag needs to be a row vector   !!!') ;
  disp('!!!   Terminating Execution                 !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;

if length(Class1Flag) ~= n ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:              !!!') ; 
  disp(['!!!   Class1Flag needs to have length ' num2str(n)]) ;
  disp('!!!   Terminating Execution                 !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;

if length(Class2Flag) ~= n ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:              !!!') ; 
  disp(['!!!   Class2Flag needs to have length ' num2str(n)]) ;
  disp('!!!   Terminating Execution                 !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;

if ~islogical(Class1Flag) ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:                  !!!') ; 
  disp('!!!   Class1Flag needs to be a logical vector   !!!') ;
  disp('!!!   Terminating Execution                     !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;

if ~islogical(Class2Flag) ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:                  !!!') ; 
  disp('!!!   Class2Flag needs to be a logical vector   !!!') ;
  disp('!!!   Terminating Execution                     !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;

if sum(Class1Flag&Class2Flag) > 0 ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from ClustIndSM.m:                           !!!') ; 
  disp('!!!   Class1Flag & Class2Flag flag a common data point   !!!') ;
  disp('!!!   Terminating Execution                              !!!') ; 
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ClustInd = [] ;
  return ;
end ;


%  Compute Cluster Index
%
if d == 1 ;    %  have only d = 1

  mdataAll = mdata(:,Class1Flag|Class2Flag) ;
      %  need to do this, since some data points may not be in either class
  totd = sum(sum((mdataAll - mean(mdataAll,2)).^2)) ;
      %  Total sums of square distance from mean of column vectors

  class1data = mdata(:,Class1Flag) ;
  class1d = sum(sum((class1data - mean(class1data,2)).^2)) ;

  class2data = mdata(:,Class2Flag) ;
  class2d = sum(sum((class2data - mean(class2data,2)).^2)) ;

else ;    %  d > 1 

  mdataAll = mdata(:,Class1Flag|Class2Flag) ;
      %  need to do this, since some data points may not be in either class
  totd = sum(sum((mdataAll - vec2matSM(mean(mdataAll,2),size(mdataAll,2))).^2)) ;
      %  Total sums of square distance from mean of column vectors

  class1data = mdata(:,Class1Flag) ;
  class1d = sum(sum((class1data - vec2matSM(mean(class1data,2),size(class1data,2))).^2)) ;

  class2data = mdata(:,Class2Flag) ;
  class2d = sum(sum((class2data - vec2matSM(mean(class2data,2),size(class2data,2))).^2)) ;

end ;

ClustInd = (class1d + class2d) / totd ;



