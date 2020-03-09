function [kde,xgrid,mker] = kdeSM(data,paramstruct) 
% KDESM, Kernel Density Estimate (1-d, Gaussian Kernel)
%   Steve Marron's matlab function
%     Does 1-d kernel density estimation, using binned (default) or 
%     direct (either matrix, or loops for bigger data sets), 
%     implementations, with the bandwidth either user specified 
%     (can be vector), or data driven (SJPI, Normal Reference, 
%     Silverman's ROT2, Oversmoothed).
%
% Inputs:
%   data        - either n x 1 column vector of 1-d data
%                     or vector of bincts, when imptyp = -1
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
%                    Version for easy copying and modification:
%    paramstruct = struct('',, ...
%                         '',, ...
%                         '',) ;
%
%    fields            values
%
%    vh               vector of bandwidths, or specifies data driven:
%                       0 (or not specified)  -  Sheather Jones Plug In
%                       -1  -  Simple Normal Reference
%                       -2  -  Silverman's Rule Of Thumb 2 
%                                 (20% Smaller than min of sd and IQR)
%                       -3  -  Oversmoothed
%                           Note: <0 only works for imptype = 0
%                       >0  -  Use input number (numbers if vector)
%                           Note: this MUST be >0 for imptyp >= 1
%                                      (the direct implementations)
%
%    vxgrid           vector of parameters for, or values of, grid to evaluate at:
%                       0 (or not specified)  -  use endpts of data and 401 bins
%                       1  -  use axisSM and 401 bins
%                       [le; lr]  -  le is left end, re is right, 401 bins
%                              (get error message and no return if le > lr)
%                       [le; lr; nb] - le left, re right, and nb bins
%                       xgrid  -  Use input values 
%                                 Note:  need to have more than 3 entries,
%                                      and only works when imptyp = 1 or 2
%
%    imptyp           flag indicating implementation type:
%                      -1  -  binned version, and "data" is assumed to be
%                                        bincounts of prebinned data
%                       0 (or not specified)  -  linear binned version
%                                        and bin data here
%                       1  -  Direct matrix implementation
%                       2  -  Slow looped implementation (only useful when
%                                 1 creates matrices that are too large)
%
%    eptflag          endpoint truncation flag (only has effect when imptyp = 0):
%                       0 (or not specified)  -  move data outside range to
%                                        nearest endpoint
%                       1  -  truncate data outside range
%
%    ibdryadj         index of boundary adjustment
%                       0 (or not specified)  -  no adjustment
%                       1  -  mirror image adjustment
%                       2  -  circular design
%
%    idatovlay        0  Do not overlay data on kde plot
%                     1  (default) overlay data using heights based on data ordering
%                              Note:  To see "c.d.f. style" increasing line, 
%                                     should also sort the data
%                     2  overlay data using random heights
%                     another integer > 0,  overlay data, using random heights,
%                                           with this number as the seed (so can 
%                                           better match data points across plots),
%                                           (should be an integer with <= 8 digits)
%
%    ndatovlay     number of data points overlayed (only has effect for idatovlay > 0)
%                       1  -  (default) overlay up to 1000 points 
%                                           (random choice, when more)
%                       2  -  overlay full data set
%                       n > 2   -  overlay n random points
%
%    datovlaymax      maximum (on [0,1] scale, with 0 at bottom, 1 at top of plot)
%                     of vertical range for overlaid data.  Default = 0.6
%
%    datovlaymin      minimum (on [0,1] scale, with 0 at bottom, 1 at top of plot)
%                     of vertical range for overlaid data.  Default = 0.5
%
%    dolcolor         data overlay color
%                     string (any of 'r', 'g', 'k', etc.) for that single color
%                               default is 'g'
%                     1  Matlab 7 color default
%                     2  time series version (ordered spectrum of colors)
%                     nx3 color matrix:  a color label for each data point
%
%    dolmarkerstr     Can be either a single string with symbol to use for marker,
%                         e.g. 'o', '.' (default), '+', 'x'
%                         (see "help plot" for a full list)
%                     Or a character array (n x 1), of these symbols,
%                         One for each data vector, created using:  strvcat
%
%    ibigdot          0  (default)  use Matlab default for dot sizes
%                     1  force large dot size in prints (useful since some
%                              postscript graphics leave dots too small)
%                              (Caution: shows up as small in Matlab view)
%
%    linewidth        width of lines (only has effect when plot is made here)
%                           default is 2, for length(vh) <= 3,
%                           default is 0.5, for length(vh) > 3,
%
%    linecolor        string with color for lines
%                                    (only has effect when plot is made here)
%                           default is 'b'
%                           use the empty string, '', for standard Matlab colors
%
%    titlestr         string with title (only has effect when plot is made here)
%                           default is empty string, '', for no title
%
%    titlefontsize    font size for title
%                                    (only has effect when plot is made here,
%                                     and when the titlestr is nonempty)
%                           default is empty [], for Matlab default
%
%    xlabelstr        string with x axis label
%                                    (only has effect when plot is made here)
%                           default is empty string, '', for no xlabel
%
%    ylabelstr        string with y axis label
%                                    (only has effect when plot is made here)
%                           default is empty string, '', for no ylabel
%
%    labelfontsize    font size for axis labels
%                                    (only has effect when plot is made here,
%                                     and when a label str is nonempty)
%                           default is empty [], for Matlab default
%
%    plotbottom      bottom of plot window,
%                                    use [] to get 0 - .05 * range (default)
%
%    plottop         top of plot window,
%                                    use [] to get max + .05 * range (default)
%
%    iplot            1  -  plot even when there is numerical output
%                     0  -  (default) only plot when no numerical output
%
% Outputs:
%     (none)  -  Draws a graph of the result (in the current axes)
%     kde     -  col vector of heights of kernel kernel density estimate,
%                    unless vh is a vector (then have matrix, with
%                    corresponding cols as density estimates)
%     xgrid   -  col vector grid of points at which estimate(s) are 
%                    evaluted (useful for plotting), unless grid is input,
%                    can also get this from linspace(le,re,nb)'  
%     mker    -  matrix (vector) of kernel functions, evaluated at xgrid,
%                    which can be plotted to show "effective window 
%                    sizes" (currently scaled to have mass 0.05, may
%                    need some vertical rescaling)
%
% Assumes path can find personal functions:
%    vec2matSM.m
%    axisSM.m
%    lbinrSM.m
%    bwsjpiSM.m
%    bwosSM.m
%    bwrotSM.m
%    bwsnrSM.m
%    iqrSM
%    cquantSM
%    bwrfphSM.m
%    rootfSM

%    Copyright (c) J. S. Marron 1996-2004




%  First set all parameters to defaults
vh = 0 ;      %  use default SJPI
vxgrid = 0 ;
imptyp = 0 ;
eptflag = 0 ;
ibdryadj = 0 ;
idatovlay = 1 ;
ndatovlay = 1 ;
datovlaymax = 0.6 ;
datovlaymin = 0.5 ;
dolcolor = 'g' ;
dolmarkerstr = '.' ;
ibigdot = 0 ;
linewidth = 2 ;
linecolor = 'b' ;
titlestr = '' ;
titlefontsize = [] ;
xlabelstr = '' ;
ylabelstr = '' ;
labelfontsize = [] ;
plotbottom = [] ;
plottop = [] ;
iplot = 0 ;


%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 1 ;   %  then paramstruct has been added

  if isfield(paramstruct,'vh') ;    %  then change to input value
    vh = getfield(paramstruct,'vh') ; 
  end ;

  if isfield(paramstruct,'vxgrid') ;    %  then change to input value
    vxgrid = getfield(paramstruct,'vxgrid') ; 
  end ;

  if isfield(paramstruct,'imptyp') ;    %  then change to input value
    imptyp = getfield(paramstruct,'imptyp') ; 
  end ;

  if isfield(paramstruct,'eptflag') ;    %  then change to input value
    eptflag = getfield(paramstruct,'eptflag') ; 
  end ;

  if isfield(paramstruct,'ibdryadj') ;    %  then change to input value
    ibdryadj = getfield(paramstruct,'ibdryadj') ; 
  end ;

  if isfield(paramstruct,'idatovlay') ;    %  then change to input value
    idatovlay = getfield(paramstruct,'idatovlay') ; 
  end ;

  if isfield(paramstruct,'ndatovlay') ;    %  then change to input value
    ndatovlay = getfield(paramstruct,'ndatovlay') ; 
  end ;

  if isfield(paramstruct,'datovlaymax') ;    %  then change to input value
    datovlaymax = getfield(paramstruct,'datovlaymax') ; 
  end ;

  if isfield(paramstruct,'datovlaymin') ;    %  then change to input value
    datovlaymin = getfield(paramstruct,'datovlaymin') ; 
  end ;

  if isfield(paramstruct,'dolcolor') ;    %  then change to input value
    dolcolor = getfield(paramstruct,'dolcolor') ; 
  end ;

  if isfield(paramstruct,'dolmarkerstr') ;    %  then change to input value
    dolmarkerstr = getfield(paramstruct,'dolmarkerstr') ; 
  end ;

  if isfield(paramstruct,'ibigdot') ;    %  then change to input value
    ibigdot = getfield(paramstruct,'ibigdot') ; 
  end ;

  if isfield(paramstruct,'linewidth') ;    %  then change to input value
    linewidth = getfield(paramstruct,'linewidth') ; 
  end ;

  if isfield(paramstruct,'linecolor') ;    %  then change to input value
    linecolor = getfield(paramstruct,'linecolor') ; 
  end ;

  if isfield(paramstruct,'titlestr') ;    %  then change to input value
    titlestr = getfield(paramstruct,'titlestr') ; 
  end ;

  if isfield(paramstruct,'titlefontsize') ;    %  then change to input value
    titlefontsize = getfield(paramstruct,'titlefontsize') ; 
  end ;

  if isfield(paramstruct,'xlabelstr') ;    %  then change to input value
    xlabelstr = getfield(paramstruct,'xlabelstr') ; 
  end ;

  if isfield(paramstruct,'ylabelstr') ;    %  then change to input value
    ylabelstr = getfield(paramstruct,'ylabelstr') ; 
  end ;

  if isfield(paramstruct,'labelfontsize') ;    %  then change to input value
    labelfontsize = getfield(paramstruct,'labelfontsize') ; 
  end ;

  if isfield(paramstruct,'plotbottom') ;    %  then change to input value
    plotbottom = getfield(paramstruct,'plotbottom') ; 
  end ;

  if isfield(paramstruct,'plottop') ;    %  then change to input value
    plottop = getfield(paramstruct,'plottop') ; 
  end ;

  if isfield(paramstruct,'iplot') ;    %  then change to input value
    iplot = getfield(paramstruct,'iplot') ; 
  end ;


end ;  %  of resetting of input parameters


if size(data,2) > 1 ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from kdeSM.m:            !!!') ;
    disp('!!!   data must be a column vector   !!!') ;
    disp('!!!   Terminating Execution          !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    return ;
end ;

n = length(data) ;

if imptyp == -1 ;

  if length(vxgrid) > 3 ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from kdeSM.m:      !!!') ;
    disp('!!!   cannot use full vxgrid,  !!!') ;
    disp('!!!   when imptyp = -1         !!!') ;
    disp('!!!   Terminating Execution    !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    return ;
  end ;

  if idatovlay ~= 0 ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from kdeSM.m:        !!!') ;
    disp('!!!   cannot have idatovlay ~= 0   !!!') ;
    disp('!!!   when imptyp = -1             !!!') ;
    disp('!!!   Resetting: idatovlay = 0     !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    idatovlay = 0 ;
  end ;

end ;


if  ~(idatovlay == 0)   &  imptyp >= 0  ;    %  then will add data to plot

  if ndatovlay == 1 ;
    ndo = min(n,1000) ;
  elseif ndatovlay == 2 ;
    ndo = n ;
  else ;
    ndo = min(n,ndatovlay) ;
  end ;

  if ~isstr(dolmarkerstr) ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from kdeSM.m:                    !!!') ;
      disp('!!!   dolmarkerstr must be of type string      !!!') ;
      disp('!!!   Resetting to default dolmarkerstr: ''.''   !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      dolmarkerstr = '.' ;
  else ;
    if ~(size(dolmarkerstr,2) == 1) ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from kdeSM.m:                    !!!') ;
      disp('!!!   dolmarkerstr must have 1 column          !!!') ;
      disp('!!!   Resetting to default dolmarkerstr: ''.''   !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      dolmarkerstr = '.' ;
    end ;
    if  ~(size(dolmarkerstr,1) == n)  &  ~(size(dolmarkerstr,1) == 1)  ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from kdeSM.m:                    !!!') ;
      disp(['!!!   dolmarkerstr must have ' num2str(n) ' rows']) ;
      disp('!!!   Resetting to default dolmarkerstr: ''.''   !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      dolmarkerstr = '.' ;
    end ;
  end ;

  %  Now check inputs for Data Over Lay Colors
  if ~isstr(dolcolor) ;

    if size(dolcolor,1) > 1 ;    %  then should have a color matrix entered
      if ~(size(dolcolor,2) == 3) ;
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
        disp('!!!   Warning from kdeSM.m:                     !!!') ;
        disp('!!!   dolcolor as a matrix must have 3 columns  !!!') ;
        disp('!!!   Resetting to default color: ''g''           !!!') ;
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
        dolcolor = 'g' ;
      elseif ~(size(dolcolor,1) == n) ;
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
        disp('!!!   Warning from kdeSM.m:                     !!!') ;
        disp(['!!!   dolcolor as a matrix must have ' num2str(n) ' rows']) ;
        disp('!!!   Resetting to default color: ''g''           !!!') ;
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
        dolcolor = 'g' ;
      end ;

    else ;
    
      if dolcolor == 1 ;    %  Use Matlab 7 color default

        kdecolor = 'k' ;
        colmap1 = [     0         0    1.0000 ; ...
                        0    0.5000         0 ; ...
                   1.0000         0         0 ; ...
                        0    0.7500    0.7500 ; ...
                   0.7500         0    0.7500 ; ...
                   0.7500    0.7500         0 ; ...
                   0.2500    0.2500    0.2500 ] ;
            %  color of projection dots, matlab default
        colmap = colmap1 ;
        while size(colmap,1) < n ;
          colmap = [colmap; colmap1] ;
        end ;
        dolcolor = colmap(1:n,:) ;

      elseif dolcolor == 2 ;    %  Use ordered spectrum of colors

        %  set up color map stuff
        %
        %  1st:    R  1      G  0 - 1    B  0
        %  2nd:    R  1 - 0  G  1        B  0
        %  3rd:    R  0      G  1        B  0 - 1
        %  4th:    R  0      G  1 - 0    B  1
        %  5th:    R  0 - 1  G  0        B  1

        nfifth = ceil((n - 1) / 5) ;
        del = 1 / nfifth ;
        vwt = (0:del:1)' ;
        colmap = [flipud(vwt), zeros(nfifth+1,1), ones(nfifth+1,1)] ;
        colmap = colmap(1:size(colmap,1)-1,:) ;
              %  cutoff last row to avoid having it twice
        colmap = [colmap; ...
                  [zeros(nfifth+1,1), vwt, ones(nfifth+1,1)]] ;
        colmap = colmap(1:size(colmap,1)-1,:) ;
              %  cutoff last row to avoid having it twice
        colmap = [colmap; ...
                  [zeros(nfifth+1,1), ones(nfifth+1,1), flipud(vwt)]] ;
        colmap = colmap(1:size(colmap,1)-1,:) ;
              %  cutoff last row to avoid having it twice
        colmap = [colmap; ...
                  [vwt, ones(nfifth+1,1), zeros(nfifth+1,1)]] ;
        colmap = colmap(1:size(colmap,1)-1,:) ;
              %  cutoff last row to avoid having it twice
        colmap = [colmap; ...
                  [ones(nfifth+1,1)], flipud(vwt), zeros(nfifth+1,1)] ;
              %  note: put this together upside down
        dolcolor = colmap(1:n,:) ;

      else ;

        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
        disp('!!!   Warning from kdeSM.m:              !!!') ;
        disp('!!!   invalid dolcolor                   !!!') ;
        disp('!!!   Resetting to default color: ''g''    !!!') ;
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
        dolcolor = 'g' ;

      end ;

    end ;

  end ;


end ;



%  Calculate kde
%
if imptyp > 0 ;    %  Then do direct implementation

  if min(vh) > 0 ;    %  Then have valid bandwidths, so proceed

    if length(vxgrid) > 3 ;  %  Then use input grid
      xgrid = vxgrid ;
      nbin = length(xgrid) ;
      lend = min(xgrid) ;
      rend = max(xgrid) ;
    else ;                    %  Need to generate a grid
      nbin = 401 ;         %  Default
      lend = min(data) ;   %  Default
      rend = max(data) ;   %  Default
      if  length(vxgrid) == 1  &  vxgrid > 0  ;      %  use AxisSM
        vax = axisSM(data) ;
        lend = vax(1) ;
        rend = vax(2) ;
      end ;
      if length(vxgrid) >= 2 ;      %  use input endpoints
        lend = vxgrid(1) ;
        rend = vxgrid(2) ;
      end ;
      if length(vxgrid) == 3 ;      %  use number of grid points
        nbin = vxgrid(3) ;
      end ;

      if lend > rend ;    %  Then bad range has been input
        disp('!!!   Error in kdeSM: invalid range chosen  !!!') ;
        xgrid = [] ;
      else ;
        xgrid = linspace(lend,rend,nbin)' ;
      end ;
    end ;


    %  do boundary adjustment if needed
    %
    if ibdryadj == 1 ;    %  then do mirror image adjustment
      badata = [(lend - (data - lend)); data; (rend + (rend - data))] ;
    elseif ibdryadj == 2 ;    %  then do circular design adjustment
      badata = [(data - (rend - lend)); data; (rend - lend + data)] ;
    else ;
      badata = data ;
    end ;


    %  Loop through bandwidths
    kde = [] ;
    for ih = 1:length(vh) ;
      h = vh(ih) ;

      if imptyp ~= 2 ;  %  Then do direct matrix implementation
        kdeh = vec2matSM((badata ./ h),nbin) - vec2matSM((xgrid' ./ h),n) ;
          %  efficient way to divide all dif's by h
          %  variable name "kde" is used to avoid creating too many biggies
        kdeh = exp(-(kdeh .^2) / 2) ;
          %  exponential part of Gaussian density
        kdeh = sum(kdeh)' ;
          %  sum part of kde, and make result a column vector
        kdeh = kdeh / (n * h * sqrt(2 * pi)) ;
          %  normalize, and mult by Gaussain density constant
        kde = [kde kdeh] ;
      else ;   %  Do slower looped implementation
        kdeh = [] ;
        for ixg = 1:nbin ;    %  Loop through grid points
          kdehx = (badata - xgrid(ixg)) / h ;
          kdehx = sum(exp(-(kdehx .^2) / 2)) ;
          kdeh = [kdeh; kdehx] ;
        end ;
        kdeh = kdeh / (n * h * sqrt(2 * pi)) ;
        kde = [kde kdeh] ;
      end ;
    end ;

  else ;    %  Have invalid bandwidths

    disp('!!!   Error in kdeSM: A bandwidth is invalid   !!!') ;
    disp('!!!   Terminating Execution                    !!!') ;
    disp('    (Note: cannot use data driven, with direct impl''s)') ;
    return ;

  end ;

else ;     %  Then do binned implementation

  if imptyp == -1 ;   %  Then data have already been binned

    if (length(vxgrid) == 1) | (length(vxgrid) > 3) ;
                         %  Then can't proceed because don't have bin ends
      disp('!!!   Error: kdeSM needs to know the endpoints   !!!') ;
      disp('!!!            to use this implementation        !!!') ;
      disp('!!!   Terminating Execution                      !!!') ;
      return ;
      bincts = [] ;
    else ;
      bincts = data ;

      nbin = 401 ;
      lend = vxgrid(1) ;
      rend = vxgrid(2) ;
      if length(vxgrid) == 3 ;          %  then use number of grid points
        nbin = vxgrid(3) ;
      end ;

      if nbin ~= length(bincts) ;    %  Then something is wrong
        disp('!!!   Warning: kdeSM was told the wrong number of bins   !!!') ;
        disp('!!!            will just use the number of counts.       !!!') ;
        nbin = size(bincts,1) ;
      end ;
    end ;

  else ;               %  Then need to bin data

    if length(vxgrid) > 3 ;  %  Then need to warn of change to default
      disp('!!!   Warning: kdeSM was given an xgrid, and also   !!!') ;
      disp('!!!       asked to bin; will bin and ignore xgrid   !!!') ;
    end ;

    %  Specify grid parameters
    nbin = 401 ;         %  Default
    lend = min(data) ;   %  Default
    rend = max(data) ;   %  Default
    if  length(vxgrid) == 1  &  vxgrid > 0  ;      %  use AxisSM
      vax = axisSM(data) ;
      lend = vax(1) ;
      rend = vax(2) ;
    end ;
    if (length(vxgrid) == 2) | (length(vxgrid) == 3) ;
                                     %  then use input endpoints
      lend = vxgrid(1) ;
      rend = vxgrid(2) ;
    end ;
    if length(vxgrid) == 3 ;          %  then use number of grid points
      nbin = vxgrid(3) ;
    end ;

    if lend > rend ;    %  Then bad range has been input
      disp('!!!   Error in kdeSM: invalid range chosen  !!!') ;
      disp('!!!   Terminating Execution                      !!!') ;
      return ;
      bincts = [] ;
    else ;
      bincts = lbinrSM(data,[lend,rend,nbin],eptflag) ;
    end ;

    %  Can do data-based bandwidth selection here, if specified
    if vh == -1 ;        %  Then use Simple Normal Reference
      vh = bwsnrSM(data) ;
    elseif vh == -2 ;    %  Then use Silverman's Rule Of Thumb 2 
                          %  (~10% Smaller than min of sd and IQR)
      vh = bwrotSM(data) ;
    elseif vh == -3 ;    %  Then use Terrell's Oversmoother
      vh = bwosSM(data) ;
    elseif min(vh) <= 0 ;     %  Then be sure to use default SJPI 
                          %    (in case an unsupported value was input)
      vh = 0 ;
    end ;

  end ;
  n = round(sum(bincts)) ;
          %  put this here in case of truncations during binning


  %  do boundary adjustment if needed
  %
  if ibdryadj == 1 ;    %  then do mirror image adjustment
    babincts = [flipud(bincts); bincts; flipud(bincts)] ;
  elseif ibdryadj == 2 ;    %  then do circular design adjustment
    babincts = [bincts; bincts; bincts] ;
  else ;
    babincts = bincts ;
  end ;


  %  Get bandwidth (if still not yet specified)
  if vh == 0 ;    %  Then use SJPI bandwidth
      vh = bwsjpiSM(bincts,[lend; rend; nbin],0,-1) ;
  end ;


  %  Loop through bandwidths
  kde = [] ;
  for ih = 1:length(vh) ;
    h = vh(ih) ;

    %  Create vector of kernel values, at equally spaced grid
    delta = (rend - lend) / (nbin - 1) ;    %  binwidth
    k = nbin - 1 ;    %  index of last nonzero entry of kernel vector
    arg = linspace(0,k * delta / h,k + 1)' ;
    kvec = exp(-(arg.^2) / 2) / sqrt(2 * pi) ;
    kvec = [flipud(kvec(2:k+1)); kvec] ;

    %  Do actual kernel density estimation
    kdeh = conv(babincts,kvec) ;

    if  ibdryadj == 1  |  ibdryadj == 2 ;    %  then did boundary adjustment
      kdeh = kdeh(nbin+k+1:k+2*nbin) / (n * h) ;
    else ;
      kdeh = kdeh(k+1:k+nbin) / (n * h) ;
    end ;

    if h < 3 * delta ;    %  Then need to normalize
                             %  to make numerical integral roughly 1
      kdeh = kdeh / (sum(kdeh) * delta) ;
    end ;

    kde = [kde kdeh] ;
  end ;

  xgrid = linspace(lend,rend,nbin)' ;

end ;



%  Create matrix of kernels, if this is needed
%
if nargout == 3 ;
  cent = mean([lend; rend]) ;
          %  centerpoint of evaluation grid
  if length(vh) > 1 ;
    mih = vec2matSM(1 ./ vh',nbin) ;
    mker = vec2matSM((xgrid - cent),length(vh)) .* mih;
          %  argument of gaussian kernel
  else ;
    mih = 1 / vh ;
    mker = (xgrid - cent) .* mih;
          %  argument of gaussian kernel
  end ;
  mker = exp(-mker.^2 / 2) .* mih / sqrt(2 * pi) ;
          %  Gaussian kernels with mass 1
  mker = 0.05 * mker ;
          %  Make masses = 0.05
end ;



%  Make plots if no numerical output requested, or if plot requested
%
if  nargout == 0  | ...
      iplot == 1  ;  %  Then make a plot


  if  length(vh) > 3  &  ~isfield(paramstruct,'linewidth')  ;
                              %  then need to change default value of linewidth
    linewidth = 0.5 ;
  end ;


  if isempty(linecolor) ;
    plot(xgrid,kde,'LineWidth',linewidth) ;
  else ;
    plot(xgrid,kde,'LineWidth',linewidth,'Color',linecolor) ;
  end ;

  if  isempty(plottop)  &  isempty(plotbottom)  ;    %  then adjust top and bottom
    plotbottom = 0 ;
    plottop = max(max(kde)) ;
    plotrange = plottop - plotbottom ;
    plotbottom = plotbottom - 0.05 * plotrange ;
    plottop = plottop + 0.05 * plotrange ;
  elseif isempty(plottop) ;                          %  then only adjust top
    plottop = max(max(kde)) ;
    plotrange = plottop - plotbottom ;
    plottop = plottop + 0.05 * plotrange ;
  elseif isempty(plotbottom) ;                       %  then only adjust bottom
    plotbottom = 0 ;
    plotrange = plottop - plotbottom ;
    plotbottom = plotbottom - 0.05 * plotrange ;
  end ;

  vax = [lend,rend,plotbottom,plottop] ;
  axis(vax) ;




  %  Set up data overlay
  %
  if  ~(idatovlay == 0)   &  imptyp >= 0  ;    %  then add data to plot

    if idatovlay > 2 ;
      rand('seed',idatovlay) ;
    end ;

    if ndo < length(data) ;    %  then need to subsample
      [temp,randind] = sort(rand(length(data),1)) ;
            %  randind is a random permutation of 1,2,...,n
      vindol = randind(1:ndo) ;
            %  indices of points to overlay
      vindol = sort(vindol) ;
            %  put back in order to preserve ordering
    else ;    %  overlay full data set
      vindol = (1:length(data))' ;
    end ;


    dataol = data(vindol) ;
    if ~isstr(dolcolor) ;
      dolcolorol = dolcolor(vindol,:) ;
    end ;
    if size(dolmarkerstr,1) > 1 ;
      dolmarkerstrol = dolmarkerstr(vindol) ;
    end ;


    flagleft = (dataol < lend) ;
        %  ones where data below left end
    flagright = (dataol > rend) ;
        %  ones where data above right end
    nleft = sum(flagleft) ;
    nright = sum(flagright) ;
    if nleft + nright > 0 ;    %  then need to deal with points outside range

      if eptflag == 1 ;    %  then truncate data outside range
        datatrunc = dataol(~(flagleft | flagright)) ;
            %  keep data that is not (outside left or outside right)
        if ~isstr(dolcolor) ;
          dolcolortrunc = dolcolorol(~(flagleft | flagright),:) ;
        end ;
        if size(dolmarkerstr,1) > 1 ;
          dolmarkerstrtrunc = dolmarkerstrol(~(flagleft | flagright)) ;
        end ;
        ndo = length(datatrunc) ;

      else ;    %  then move outside points to nearest end
        datatrunc = dataol ;
        if ~isstr(dolcolor) ;
          dolcolortrunc = dolcolorol ;
        end ;
        if size(dolmarkerstr,1) > 1 ;
          dolmarkerstrtrunc = dolmarkerstrol ;
        end ;
        if nleft > 0 ;    %  then replace those points with lend
          datatrunc(flagleft) = lend * ones(nleft,1) ;
        end ;
        if nright > 0 ;    %  then replace those points with rend
          datatrunc(flagright) = rend * ones(nright,1) ;
        end ;

      end ;

    else ;

      datatrunc = dataol ;
      if ~isstr(dolcolor) ;
        dolcolortrunc = dolcolorol ;
      end ;
      if size(dolmarkerstr,1) > 1 ;
        dolmarkerstrtrunc = dolmarkerstrol ;
      end ;

    end ;


    if idatovlay == 1 ;    %  then take heights to be natural ordering
      hts = (datovlaymin + (datovlaymax - datovlaymin) ...
                                       * (0.5:ndo)' / ndo) * vax(4) ;
    else ;    %  then use a random ordering
      hts = (datovlaymin + (datovlaymax - datovlaymin) ...
                                       * rand(ndo,1)) * vax(4) ;
        %  random heights
    end ;


    %  overlay selected data
    %
    vax = axis ;
    hold on ;

      if  isstr(dolcolor)  &  (size(dolmarkerstr,1) == 1)  ;    %  then can plot points all together
        if ibigdot == 1 ;   %  plot deliberately large dots
          plot(datatrunc,hts,[dolcolor 'o'],'MarkerSize',1,'LineWidth',2) ;
        else ;    %  use matlab default dots
          plot(datatrunc,hts,[dolcolor dolmarkerstr]) ;
        end ;
      elseif isstr(dolcolor) ;    %  need to plot points individually, but use string color
        for idol = 1:length(datatrunc) ;
          if ibigdot == 1 ;   %  plot deliberately large dots
            plot(datatrunc(idol),hts(idol),[dolcolor 'o'],'MarkerSize',1,'LineWidth',2) ;
          else ;    %  use markers
            plot(datatrunc(idol),hts(idol),dolmarkerstrtrunc(idol),'Color',dolcolor) ;
          end ;
        end ;
      else ;    %  need to plot points individually
        for idol = 1:length(datatrunc) ;
          if ibigdot == 1 ;   %  plot deliberately large dots
            plot(datatrunc(idol),hts(idol),'o','MarkerSize',1,'LineWidth',2,'Color',dolcolortrunc(idol,:)) ;
          else ;    %  use markers
            if (size(dolmarkerstr,1) == 1) ;
              plot(datatrunc(idol),hts(idol),dolmarkerstr(1),'Color',dolcolortrunc(idol,:)) ;
            else ;
              plot(datatrunc(idol),hts(idol),dolmarkerstrtrunc(idol),'Color',dolcolortrunc(idol,:)) ;
            end ;
          end ;
        end ;
      end ;

    hold off ;


    %  reorder, so that overlaid data appears under the smooth(s)
    %
    vachil = get(gca,'Children') ;
    nchil = length(vachil) ;
    vachil = [vachil(2:nchil); vachil(1)] ;
    set(gca,'Children',vachil) ;
    

  end ;


  if ~isempty(titlestr) ;
    if isempty(titlefontsize) ;
      title(titlestr) ;
    else ;
      title(titlestr,'FontSize',titlefontsize) ;
    end ;
  end ;


  if ~isempty(xlabelstr) ;
    if isempty(labelfontsize) ;
      xlabel(xlabelstr) ;
    else ;
      xlabel(xlabelstr,'FontSize',labelfontsize) ;
    end ;
  end ;


  if ~isempty(ylabelstr) ;
    if isempty(labelfontsize) ;
      ylabel(ylabelstr) ;
    else ;
      ylabel(ylabelstr,'FontSize',labelfontsize) ;
    end ;
  end ;


end ;

