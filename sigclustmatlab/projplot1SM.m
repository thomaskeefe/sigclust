function projplot1SM(data,vdir,paramstruct) 
% PROJPLOT1SM, PROJection PLOT in 1 dimension, 
%   Steve Marron's matlab function
%     Studies one dimensional projections of data,
%     using jitter-type plots and kernel density estimates
%     based on a given direction vector
%
% Inputs:
%   data    - d x n matrix of data, each column vector is 
%                    a "d-dim digitized curve"
%
%   vdir    - a d dimensional "direction vector"
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
%     paramstruct = struct('',, ...
%                          '',, ...
%                          '',) ;
%
%    fields            values
%
%    icolor           0  fully black and white version (everywhere)
%                     string (any of 'r', 'g', 'b', etc.) that single color
%                     1  (default)  color version (Matlab 7 color default)
%                     2  time series version (ordered spectrum of colors)
%                     3  black and white time series version 
%                             (ordered spectrum of gray levels)
%                     nx3 color matrix:  a color label for each data point
%
%    markerstr        Can be either a single string with symbol to use for marker,
%                         e.g. 'o' (default), '.', '+', 'x'
%                         (see "help plot" for a full list)
%                     Or a character array (n x 1), of these symbols,
%                         One for each data vector, created using:  strvcat
%
%    isubpopkde       0  (default) construct kde using only the full data set
%                     1  partition data into subpopulations, using the color
%                            indicators in icolor (defaults to 0, unless icolor
%                            is an nx3 color matrix), as markers of subsets.
%                            The corresponding mixture colors are then used in
%                            the subdensity plot, and overlaid with the full 
%                            density shown in black
%                     2  Show only the component densities (in corresponding 
%                            colors), without showing the full population
%                            density
%
%    ibigdot          0  (default)  use Matlab default for dot sizes
%                     1  force large dot size in prints (useful since some
%                              postscript graphics leave dots too small)
%                              (Caution: shows up as small in Matlab view)
%                              Only has effect when markerstr = '.' 
%
%    idatovlay        0  Do not overlay data on kde plot
%                     1  (default) overlay data using heights based on data ordering
%                              Note:  To see "c.d.f. style" increasing line, 
%                                     should first sort the data
%                     2  overlay data using random heights
%                     another integer > 0,  overlay data, using random heights,
%                                           with this number as the seed (so can 
%                                           better match data points across plots),
%                                           (should be an integer with <= 8 digits)
%
%    ndatovlay     number of data points overlayed (only has effect for idatovlay > 0)
%                       1  -  overlay up to 1000 points 
%                                           (random choice, when more)
%                       2  -  (default) overlay full data set
%                       n > 2   -  overlay n random points
%
%    datovlaymax      maximum (on [0,1] scale, with 0 at bottom, 1 at top of plot)
%                     of vertical range for overlaid data.  Default = 0.6
%
%    datovlaymin      minimum (on [0,1] scale, with 0 at bottom, 1 at top of plot)
%                     of vertical range for overlaid data.  Default = 0.5
%
%    legendcellstr    cell array of strings for legend (nl of them),
%                     useful for (colored) classes, create this using
%                     cellstr, or {{string1 string2 ...}}
%                     CAUTION:  If are updating this field, using a command like:
%                         paramstruct = setfield(paramstruct,'legendcellstr',...
%                     Then should only use single braces in the definition of
%                     legendecellstr, i. e. {string1 string2 ...}
%                     Also a way to add a "title" to the first plot
%                             for this, use input of form:  {{string}}
%                     Also can indicate symbols, by just adding (at least 
%                             for +,x.o) into the text
%
%    mlegendcolor     nl x 3 color matrix, corresponding to cell legends above
%                     (not needed when legendcellstr not specified)
%                     (defaults to black when not specified)
%
%    vaxlim        Vector of axis limits
%                        Use [] for default of all automatically chosen, by axisSM
%                        Use 1 for symmetrically chosen, by axisSM
%                            (often preferred for centered plots, as in PCA)
%                                 Caution:  For either of these, projections 
%                                           which are all the same will 
%                                           generate a warning message, and
%                                           be shown using common value +- 1
%                                           and bandwidth of 0.1
%                        Otherwise, must be 1 x 4 or 1 x 2 row vector of axis limits
%                            1 x 4 determines all 4 axis limits
%                            1 x 2 determines horizontal axis limits
%                        Note:  Use is generally not recommended,
%                        because defaults give "good visual impression
%                        of decomposition".  It is mostly intended for use 
%                        with certain types of zooming
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
%    ifigure          index for figure number:
%                       0    (default) Ignore Figure, and put in current axes
%                                (for use as component of larger scale plot)
%                       > 0  Put in Figure ifigure, first clearing the figure
%                                (usually best for stand alone plot)
%                       < 0  Put in Figure -ifigure, without clearing
%
%    savestr          string controlling saving of output,
%                         either a full path, or a file prefix to
%                         save in matlab's current directory
%                         Will add .ps, and save as either
%                             color postscript (icolor is neither 0 nor 3)
%                         or
%                             black&white postscript (when icolor = 0 or 3)
%                         unspecified:  results only appear on screen
%                     Note:  when savestr is nonempty, and ifigure = 0,
%                            give warning and reset ifigure to 1
%
%    iscreenwrite     0  (default)  no screen writes
%                     1  write to screen to show progress
%
%         These parameters create data for use by Marc Niethammer's mexplorer
%         They require savestr, and either icolor or markerstr to be manually set
%         Also, all but cellsubtypes must be non-empty
%         Then they create a figure file, which is used by mexplorer
%
%    celltypes 
%    cellsubtypes
%    slidenames
%    slideids 
%
%
%
% Outputs:
%     Graphics in current Figure
%     When savestr exists,
%        Postscript files saved in 'savestr'.ps
%                 (color postscript for icolor ~= 0)
%                 (B & W postscript for icolor = 0)
%
%
% Assumes path can find personal functions:
%    axisSM.m
%    bwsjpiSM.m
%    kdeSM.m
%    lbinrSM.m
%    vec2matSM.m
%    bwrfphSM.m
%    bwosSM.m
%    rootfSM
%    vec2matSM.m
%    bwrotSM.m
%    bwsnrSM.m
%    iqrSM.m
%    cquantSM.m


%    Copyright (c) J. S. Marron 2004-2013



%  First set all parameters to defaults
%
icolor = 1 ;
markerstr = 'o' ;
isubpopkde = 0 ;
ibigdot = 0 ;
idatovlay = 1 ;
ndatovlay = 2 ;
datovlaymax = 0.6 ;
datovlaymin = 0.5 ;
legendcellstr = {} ;
mlegendcolor = [] ;
vaxlim = [] ;
titlestr = '' ;
titlefontsize = [] ;
xlabelstr = '' ;
ylabelstr = '' ;
labelfontsize = [] ;
ifigure = 0 ;
savestr = [] ;
iscreenwrite = 0 ;

celltypes = [];
cellsubtypes = [];
slidenames = [];
slideids = [];

%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 2 ;   %  then paramstruct is an argument

  if isfield(paramstruct,'icolor') ;    %  then change to input value
    icolor = getfield(paramstruct,'icolor') ; 
  end ;

  if isfield(paramstruct,'markerstr') ;    %  then change to input value
    markerstr = getfield(paramstruct,'markerstr') ; 
  end ;

  if isfield(paramstruct,'isubpopkde') ;    %  then change to input value
    isubpopkde = getfield(paramstruct,'isubpopkde') ; 
  end ;

  if isfield(paramstruct,'ibigdot') ;    %  then change to input value
    ibigdot = getfield(paramstruct,'ibigdot') ; 
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

  if isfield(paramstruct,'legendcellstr') ;    %  then change to input value
    legendcellstr = getfield(paramstruct,'legendcellstr') ; 
  end ;

  if isfield(paramstruct,'mlegendcolor') ;    %  then change to input value
    mlegendcolor = getfield(paramstruct,'mlegendcolor') ; 
  end ;

  if isfield(paramstruct,'vaxlim') ;    %  then change to input value
    vaxlim = getfield(paramstruct,'vaxlim') ; 
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

  if isfield(paramstruct,'ifigure') ;    %  then change to input value
    ifigure = getfield(paramstruct,'ifigure') ; 
  end ;
  
  if isfield(paramstruct,'savestr') ;    %  then use input value
    savestr = getfield(paramstruct,'savestr') ; 
    if ~(ischar(savestr) | isempty(savestr)) ;    %  then invalid input, so give warning
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from projplot1SM.m:  !!!') ;
      disp('!!!   Invalid savestr,             !!!') ;
      disp('!!!   using default of no save     !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      savestr = [] ;
    end ;
  end ;

  if isfield(paramstruct,'iscreenwrite') ;    %  then change to input value
    iscreenwrite = getfield(paramstruct,'iscreenwrite') ; 
  end ;


  if isfield(paramstruct,'celltypes' );  % then change to input value
    celltypes = paramstruct.celltypes;
  end ;
  
  if isfield(paramstruct,'cellsubtypes' );  % then change to input value
    cellsubtypes = paramstruct.cellsubtypes;
  end ;
  
  if isfield(paramstruct,'slidenames' );  % then change to input value
    slidenames = paramstruct.slidenames;
  end ;
  
  if isfield(paramstruct,'slideids' );  % then change to input value
    slideids = paramstruct.slideids;
  end ;


end ;    %  of resetting of input parameters


%  Set up output for mexplorer
%  i.e.  set flag to augment with user data
%
if ( ~isempty( slidenames ) & ~isempty( slideids ) & ~isempty( celltypes ) )
  augmentWithUserData = 1;
  if ( isempty( cellsubtypes ) )
    cellsubtypes = cell( size( celltypes ) ); % just initialize it empty
  end
else
  augmentWithUserData = 0;
end



%  set preliminary stuff
%
d = size(data,1) ;
         %  dimension of each data curve
n = size(data,2) ;
         %  number of data curves

if ~(d == size(vdir,1)) ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from projplot1SM.m:   !!!') ;
  disp('!!!   Dimension of vdir must be   !!!') ;
  disp('!!!   same as dimension of data   !!!') ;
  disp('!!!   Terminating execution       !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;

if ~(1 == size(vdir,2)) ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from projplot1SM.m:   !!!') ;
  disp('!!!   vdir must be a single       !!!') ;
  disp('!!!   column vector               !!!') ;
  disp('!!!   Terminating execution       !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;

if  (size(icolor,1) > 1)  |  (size(icolor,2) > 1)  ;    %  if have color matrix
  if ~(3 == size(icolor,2)) ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:               !!!') ;
    disp('!!!   icolor as a matrix must have 3 columns    !!!') ;
    disp('!!!   Resetting to icolor = 1, Matlab default   !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    icolor = 1 ;
  elseif ~(n == size(icolor,1)) ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:               !!!') ;
    disp(['!!!   icolor as a matrix must have ' num2str(n) ' rows']) ;
    disp('!!!   Resetting to icolor = 1, Matlab default   !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    icolor = 1 ;
  end ;
end ;


if  (isubpopkde == 1)  |  (isubpopkde == 2) ;    
                     %  Then have requested subpop kdes
  if ~(size(icolor,2) == 3) ;    
                     %  But have invalid color matrix for subpop kdes
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:    !!!') ;
    disp(['!!!   isubpopkde = ' num2str(isubpopkde) ', but icolor     !!!']) ;
    disp('!!!   is not a valid color matrix    !!!') ;
    disp('!!!   Resetting isubpopkde to 0      !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    isubpopkde = 0 ;
  end ;
end ;

if  ~isempty(vaxlim)  &  ~(vaxlim == 1)  ;
  if  (size(vaxlim,1) ~= 1)  |  ...
      (  ~((size(vaxlim,2) == 2) | ...
           (size(vaxlim,2) == 4) ))  | ...
      (size(vaxlim,2) == 1)  ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from projplot1SM.m:   !!!') ;
    disp('!!!   vaxlim must be empty, = 1   !!!') ;
    disp('!!!   or a 1 x 2 row vector       !!!') ;
    disp('!!!   or a 1 x 4 row vector       !!!') ;
    disp('!!!   Terminating execution       !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    return ;
  end ;
end ;

if  ~isempty(savestr) & ifigure == 0 ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Warning from projplot1SM.m:    !!!') ;
  disp('!!!   savestr ~= [], and ifigure = 0  !!!') ;
  disp('!!!   are inconsistent,              !!!') ;
  disp('!!!   Resetting ifigure to 1         !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  ifigure = 1 ;
end ;

if  ~isempty(mlegendcolor) ;
  if ~(size(legendcellstr,2) == size(mlegendcolor,1)) ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:       !!!') ;
    disp('!!!   legendcellstr &  mlegendcolor     !!!') ;
    disp('!!!   must have the same number         !!!') ;
    disp('!!!   of entries                        !!!') ;
    disp('!!!   Note: this could be caused by     !!!') ;
    disp('!!!   using "setfield(paramstruct..."   !!!') ;
    disp('!!!   with "lengedcellstr" defined      !!!') ;
    disp('!!!   by double braces                  !!!') ;
    disp('!!!   of entries                        !!!') ;
    disp('!!!   Resetting to no legend            !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    legendcellstr = [] ;
  end ;
end ;

if  ~isempty(mlegendcolor) ;
  if ~(size(mlegendcolor,2) == 3) ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:   !!!') ;
    disp('!!!   mlegendcolor                  !!!') ;
    disp('!!!   must have 3 columns           !!!') ;
    disp('!!!   Resetting to no legend        !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    legendcellstr = [] ;
  end ;
end ;

if  size(markerstr,1) == 1  &  size(markerstr,2) == 1  ;
  if  (ibigdot == 1)  &  ~strcmp(markerstr,'.')  ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:   !!!') ;
    disp('!!!   ibigdot is set to 1,          !!!') ;
    disp('!!!   but a non-dot appears         !!!') ;
    disp('!!!   in markerstr                  !!!') ;
    disp('!!!   Resetting to ibigdot = 0      !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    ibigdot = 0 ;
  end ;
else ;
  if ~(size(markerstr,1) == n  & ...
       size(markerstr,2) == 1  & ...
       ischar(markerstr)) ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from projplot1SM.m:             !!!') ;
    disp('!!!   markerstr must be a character array   !!!') ;
    disp('!!!   with n rows and one column            !!!') ;
    disp(['!!!   Size of input dat is:   ' num2str(size(data))]) ;
    disp(['!!!   Size of markerstr is:   ' num2str(size(markerstr))]) ;
    disp('!!!   Terminating execution                 !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    return ;
  end ;
  if ibigdot == 1 ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:   !!!') ;
    disp('!!!   ibigdot is set to 1,          !!!') ;
    disp('!!!   but markerstr has             !!!') ;
    disp('!!!   multiple entries              !!!') ;
    disp('!!!   Resetting to ibigdot = 0      !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    ibigdot = 0 ;
  end ;
end ;



%  Set up appropriate colors
%
if  size(icolor,1) == 1   &  size(icolor,2) == 1 ;    %  then have scalar input

  if icolor == 0 ;    %  then do everything black and white

    kdecolor = 'k' ;
    dotcolor = 'k' ;
        %  color of projection dots

    indivplotflag = 0 ;
    icolorprint = 0 ;

  elseif ischar(icolor) ;

    kdecolor = 'k' ;
    dotcolor = icolor ;
        %  string for color of projection dots
    indivplotflag = 0 ;
    icolorprint = 1 ;

  elseif icolor == 1 ;    %  then do MATLAB 7 color default

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
    colmap = colmap(1:n,:) ;

    indivplotflag = 1 ;
    icolorprint = 1 ;


  elseif icolor == 2 ;    %  then do spectrum for ordered time series

    kdecolor = 'k' ;

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

    indivplotflag = 1 ;
    icolorprint = 1 ;

  elseif icolor == 3 ;    %  then do gray levels for ordered time series

    kdecolor = 'k' ;

    %  set up color map stuff
    %
    startgray = 0.9 ;
    vgray = linspace(startgray,0,n) ;
    colmap = vgray' * ones(1,3) ;

    indivplotflag = 1 ;
    icolorprint = 1 ;
        %  Leave this as "color" to get gray levels to show up in .ps file

  end ;


elseif  size(icolor,2) == 3  ;    %  then have valid color matrix

  kdecolor = 'k' ;
  colmap = icolor ;

  indivplotflag = 1 ;
  icolorprint = 1 ;

else ;    %   invalid color matrix input

  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from projplot1SM.m:           !!!') ;
  disp('!!!   Invalid icolor input,               !!!') ;
  disp('!!!   must be a scalar, or color matrix   !!!') ;
  disp('!!!   Terminating execution               !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;

end ;



%  Set up data overlay
%
if ~(idatovlay == 0) ;    %  then add data to plot

  if ndatovlay == 1 ;
    ndo = min(n,1000) ;
  elseif ndatovlay == 2 ;
    ndo = n ;
  else ;
    ndo = min(n,ndatovlay) ;
  end ;

  if ndo < n ;    %  then need to subsample
    [temp,randperm] = sort(rand(n,1)) ;
          %  randperm is a random permutation of 1,2,...,n
    vindol = randperm(1:ndo) ;
          %  indices of points to overlay
    vindol = sort(vindol) ;
          %  put back in order to preserve ordering
  else ;    %  overlay full data set
    vindol = (1:n)' ;
  end ;

else ;
  ndo = 0 ;
end ;



%  Set up for multiple markers if (needed)
%
if size(markerstr,1) > 1;    %  then have already input full
                             %  character array, so use it
  mmarks = markerstr ;
  if indivplotflag == 0 ;    %  then need to reset this 
                             %  and create a colmap

    indivplotflag = 1 ;
    if strcmp(dotcolor,'k') ;
      vcolor = [0 0 0] ;
    elseif strcmp(dotcolor,'r') ;
      vcolor = [1 0 0] ;
    elseif strcmp(dotcolor,'g') ;
      vcolor = [0 1 0] ;
    elseif strcmp(dotcolor,'b') ;
      vcolor = [0 0 1] ;
    elseif strcmp(dotcolor,'c') ;
      vcolor = [0 1 1] ;
    elseif strcmp(dotcolor,'m') ;
      vcolor = [1 0 1] ;
    elseif strcmp(dotcolor,'y') ;
      vcolor = [1 1 0] ;
    elseif strcmp(dotcolor,'w') ;
      vcolor = [1 1 1] ;
    end ;
    colmap = ones(n,1) * vcolor ;
  end ;
else;    %  then are using default, or have entered single symbol
  if indivplotflag == 1 ;    %  the need to create full char array
    mmarks = [] ;
    for i=1:n ;
      mmarks = strvcat(mmarks,markerstr) ;
    end ;
  end ;
end ;


if ~isempty(legendcellstr) ;
  nlegend = length(legendcellstr) ;
  if isempty(mlegendcolor) ;
    mlegendcolor = vec2matSM(zeros(1,3),nlegend) ;
        %  all black when unspecified
  end ;
end ;




%  Compute Projections
%
vdirlen2 = sum(vdir.^2) ;
if abs(vdirlen2 - 1) < 10^(-10) ;    %  already has length 1
  vdirn = vdir ;
else ;    %  need to adjust length to be 1
  if iscreenwrite == 1 ;    %  then give a warning about adjustment
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from projplot1SM.m:          !!!') ;
    disp('!!!   vdir should be a direction vector,   !!!') ;
    disp('!!!   will adjust to length 1              !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  end ;
  if vdirlen2 < 10^(-10)  ;    %  then can't proceed, so quit
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from projplot1SM.m:   !!!') ;
    disp('!!!   vdir length is too short,   !!!') ;
    disp('!!!   Terminating execution       !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    return ;
  end ;
  vdirn = vdir / sqrt(vdirlen2) ;
end ;
vproj = vdirn' * data ;
    %  row vector of inner products with data
vproj = vproj' ;
    %  turn into a column vector




%  Set Axes & Compute Kernel Density Estimate
%
projrange = max(vproj) - min(vproj) ;
if projrange <= 0 ;    %  all projectuons are the same (i.e. 0-var variable)
  if isempty(vaxlim) ;    %  then use value +- 1
    vax = [(vproj(1) - 1) (vproj(1) + 1)] ;
    hkde = 0.1 ;
  elseif vaxlim == 1 ;    %  then use symmetrized defaults
    vax = [-(abs(vproj(1)) + 1) (abs(vproj(1)) + 1)] ;
    hkde = 0.1 ;
  else ;
    vax = vaxlim([1 2]) ;
    hkde = (vaxlim(2) - vaxlim(1)) / 20 ;
  end ;
else ;    %  have soe different projections
  if isempty(vaxlim) ;    %  then use axisSM defaults
    vax = axisSM(vproj) ;
  elseif vaxlim == 1 ;    %  then use symmetrized defaults
    vax = axisSM(vproj) ;
    vaxx = max(abs(vax([1 2]))) ;
    vax = [-vaxx vaxx] ;
  else ;
    vax = vaxlim([1 2]) ;
  end ;
    paramstruct = struct('xgridp',vax) ;
  hkde = bwsjpiSM(vproj,paramstruct) ;
end; 

  paramstruct = struct('vh',hkde,...
                       'vxgrid',vax) ;
[kde,kdexgrid] = kdeSM(vproj,paramstruct) ;

if  isempty(vaxlim)  |  ...
    vaxlim == 1  |  ...
    size(vaxlim,2) == 2  ;    %  then use axisSM default for kde
  vaxkde = axisSM(kde) ;
  vax = [vax 0 vaxkde(2)] ;
else ;
  vax = [vax vaxlim([3 4])] ;
end ;



%  Find Sub-populations and Compute KDEs (if needed)
%
if  (isubpopkde == 1)  |  (isubpopkde == 2) ;

  %  First need to identify subpopulations,
  %  according to colors found in icolor
  %
  nsubpop = 1 ;
  subpopcolor = icolor(1,:) ;
  vidatsp = 1 ;
  for i = 2:n ;
    vcolor = icolor(i,:) ;

    mflag = (vec2matSM(vcolor,nsubpop)  ==  subpopcolor) ;
        %  one where the entry of vcolor is the same as subpopcolor
    vflag = (sum(mflag,2) == 3) ;
        %  one where all three entries in the row are the same

    if sum(vflag) == 0 ;    %  then have a new color
      nsubpop = nsubpop + 1 ;
      subpopcolor = [subpopcolor; vcolor] ;
      vidatsp = [vidatsp; nsubpop] ;
    else ;    %  then have an existing color
      [temp, isubpop] = max(vflag) ;
          %  get index of place where have a one in vflag
      vidatsp = [vidatsp; isubpop] ;
    end ;

  end ;


  %  Next compute subpop KDEs
  %
  paramstruct = struct('vh',hkde,...
                       'vxgrid',vax([1 2])) ;
                           %  use same bandwidth as for full population
  mspkde = [] ;
  for isp = 1:nsubpop ;
    spflag = (vidatsp == isp) ;
        %  one where data points are in this subpopulation
    spdat = vproj(spflag) ;
        %  subpopulation data for this subpopulation
    ndatsp = sum(spflag) ;
        %  number of data points in this subpopulation
    spkde = kdeSM(spdat,paramstruct) ;
        %  mass one kde for this subpopulation
    spkde = (ndatsp / n) * spkde ;
        %  rescale, so mass is proportional to size of subpopulation
    mspkde = [mspkde spkde] ; 
  end ;


end ;    %  of subpopluation KDE if block



%  make main graphic
%
if iscreenwrite == 1 ;
  disp('  Making 1d Projection Plot') ;
end ;

if ifigure > 0 ;
  figure(ifigure) ;
  clf ;
elseif ifigure < 0 ;
  figure(-ifigure) ;
end ;


if indivplotflag == 0 ;    %  then can plot everything with a single plot call

  plot(kdexgrid,kde,[kdecolor '-']) ;
    axis(vax) ;
    hold on ;
      if ndo > 0 ;    %  then add data to plot

        if idatovlay == 1 ;    %  then take heights to be natural ordering
          hts = (datovlaymin + (datovlaymax - datovlaymin) ...
                                           * (0.5:ndo)' / ndo) * vax(4) ;
        else ;    %  then use a random ordering
          if ~(idatovlay == 2) ;
            rand('seed',idatovlay) ;
          end ;
          hts = (datovlaymin + (datovlaymax - datovlaymin) ...
                                           * rand(ndo,1)) * vax(4) ;
            %  random heights
        end ;

        if ibigdot == 1 ;   %  plot deliberately large dots
          plot(vproj(vindol),hts,[dotcolor 'o'],'MarkerSize',1,'LineWidth',2) ;
        else ;    %  use input marker
          plot(vproj(vindol),hts,[dotcolor markerstr]) ;
        end ;

      end ;


      if ~isempty(legendcellstr) ;    %  then add legend
        tx = vax(1) + 0.1 * (vax(2) - vax(1)) ;
        for ilegend = 1:nlegend ;
          ty = 0 + ((nlegend - ilegend + 1) / ...
                               (nlegend + 1)) * (vax(4) - 0) ;
          text(tx,ty,legendcellstr(ilegend),  ...
                    'Color',mlegendcolor(ilegend,:)) ;
        end ;
      end ;

    hold off ;


elseif indivplotflag == 1 ;    %  then need to do individual plot calls
                               %  for each data point

  if ~(isubpopkde == 2) ;    % then plot full data KDE
    plot(kdexgrid,kde,[kdecolor '-']) ;
  end ;
  if  (isubpopkde == 1)  |  (isubpopkde == 2) ;    %  then plot subpop KDEs
    hold on ;
      for isp = 1:nsubpop ;
        plot(kdexgrid,mspkde(:,isp),'-','Color',subpopcolor(isp,:)) ;
      end ;
    hold off ;
  end ;
    axis(vax) ;
    hold on ;
      if ndo > 0 ;    %  then add data to plot

        if idatovlay == 1 ;    %  then take heights to be natural ordering
          hts = (datovlaymin + (datovlaymax - datovlaymin) ...
                                           * (0.5:ndo)' / ndo) * vax(4) ;
        else ;    %  then use a random ordering
          if ~(idatovlay == 2) ;
            rand('seed',idatovlay) ;
          end ;
          hts = (datovlaymin + (datovlaymax - datovlaymin) ...
                                           * rand(ndo,1)) * vax(4) ;
            %  random heights
        end ;

        for idato = 1:ndo ;
          if ibigdot == 1 ;   %  plot deliberately large dots
            hC = plot(vproj(vindol(idato)),hts(vindol(idato)), ...
                        'o','Color',colmap(vindol(idato),:), ...
                        'MarkerSize',1,'LineWidth',2) ;
            if ( augmentWithUserData )
              set(hC,'UserData', struct( 'celltype', celltypes{idato}, 'cellsubtype', cellsubtypes{idato}, 'slidename', slidenames{idato}, 'slidenr', slideids(idato), 'marker', 'o', 'color', colmap(vindol(idato),:) ) );
            end
            
          else ;    %  use input marker
            hC = plot(vproj(vindol(idato)),hts(idato), ...
                         mmarks(vindol(idato)), ...
                         'Color',colmap(vindol(idato),:)) ;
            if ( augmentWithUserData )
              set(hC,'UserData', struct( 'celltype', celltypes{idato}, 'cellsubtype', cellsubtypes{idato}, 'slidename', slidenames{idato}, 'slidenr', slideids(idato), 'marker', mmarks(vindol(idato)), 'color', colmap(vindol(idato),:) ) );
            end

          end ;
        end ;

      end ;


      if ~isempty(legendcellstr) ;    %  then add legend
        tx = vax(1) + 0.1 * (vax(2) - vax(1)) ;
        for ilegend = 1:nlegend ;
          ty = 0 + ((nlegend - ilegend + 1) / ...
                               (nlegend + 1)) * (vax(4) - 0) ;
          text(tx,ty,legendcellstr(ilegend),  ...
                    'Color',mlegendcolor(ilegend,:)) ;
        end ;
      end ;

    hold off ;


end ;    %  of indivplotflag if block

%  Add title and labels
%
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



%  Save output (if needed)
%
if ~isempty(savestr) ;   %  then create postscript file

  orient landscape ;

  if icolorprint ~= 0 ;     %  then make color postscript
    print('-dpsc',savestr) ;
  else ;                %  then make black and white
    print('-dps',savestr) ;
  end ;

  if ( augmentWithUserData )
    saveas(gcf, savestr, 'fig')
  end ;

end ;




