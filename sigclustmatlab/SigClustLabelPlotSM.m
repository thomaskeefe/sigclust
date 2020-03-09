function SigClustLabelPlotSM(data,vclass,paramstruct) 
% SIGCLUSTLABELPLOTSM, statistical SIGnificance of CLUSTers,
%         does cluster LABELled PLOT(s) of data 
%   Steve Marron's matlab function
%     Shows graphics to visualize the results of a
%     2 class clustering.
%     Allows arbitrary dimension of data:
%         1-d  -  standard 1-d graphic 
%         2-d  -  regular scatterplot or 
%                     mean difference direction & ortho
%         3 or 4-d  -  regular scatterplot matrix or 
%                          mean difference direction & ortho PC
%         >4-d  -  regular PCA projections or 
%                          mean difference direction & ortho PCs
%
% Inputs:
%   data    - d x n matrix of data, each column vector is 
%                    a "d-dim data vector"
%
%   vclass  - Cluster Labelling  
%                    1 x n vector of labelings, i.e. 1's and 2's  
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
%    iMDdir           indicator for use of Mean Difference direction
%                         0 to use ordinary directions
%                               (regular for d = 2-4, PC otherwise)
%                         1 to use Mean Difference and orthogonal PC directions
%                               (default)
%                           (no effect for d = 1)
%
%    icolor           0  fully black and white version (everywhere)
%                     2x3 color matrix:  a color label for each class
%                         [[1 0 0]; [0 0 1]] is the default
%
%    markerstr        Character array (2 x 1), of symbols,
%                         e.g. 'o', '.', '+', 'x','*'
%                         created using:  strvcat
%                             e.g. strvcat('o','+') (default)
%
%    legendcellstr    cell array of strings for legend (nl of them),
%                     useful for (colored) classes, create this using
%                     cellstr, or {{string1 string2 ...}}
%                         Note:  These strange double brackets seems to be needed
%                                for correct pass to subroutine
%                                It may change in later versions of Matlab
%                     CAUTION:  If are updating this field, using a command like:
%                         paramstruct = setfield(paramstruct,'legendcellstr',...
%                     Then should only use single braces in the definition of
%                     legendecellstr, i. e. {string1 string2 ...}
%                     Also a way to add a "title" to the first plot
%                             for this, use input of form:  {{string}}
%                     Also can indicate symbols, by just adding (at least 
%                             for +,x.o) into the text
%                     Note:  this only appears on the first plot
%
%    mlegendcolor     nl x 3 color matrix, corresponding to cell legends above
%                     (not needed when legendcellstr not specified)
%                     (defaults to icolor when not specified, and nl = 2)
%                     (otherwise defaults to black when not specified)
%
%    maxlim        Matrix of axis limits
%                        Use [] for default of all automatically chosen, by axisSM
%                        Use 1 for symmetrically chosen, by axisSM
%                            (often preferred for centered plots, as in PCA)
%                        Otherwise, must be (size(mdir,2) + npcadir) x 2 
%                            matrix of axis limits, 
%                            with each row corresponding to a direction
%                        Note:  Use is generally not recommended,
%                        because defaults give "good visual impression
%                        of decomposition.  It is mostly intended to allow
%                        the highlighting of "visually different scales" in data.
%                        But this comes at the cost of reduced detail being visible in the plots
%
%    iplotaxes        0 do not plot axes
%                     1 (default) plot axes as determined by direction vectors, 
%                           using thin line type
%
%    iplotdirvec      0 (default) do not plot direction vectors
%                     1 plot direction vectors, using thick line type
%
%    ibelowdiag       0 leave off scatterplots below diagonal
%                     1 (default) show scatterplots both above and below diagonal
%
%    titlecellstr     cell array for making subplot titles
%                     default is an empty cell array, {} for no titles
%                     To add titles, this should be a cell array of vertical
%                     cellstrs, where each cell location corresponds to a subplot
%                     Can create this using a command like:
%                              {{strvcat('Title Top Left','Title Bottom Left') ...
%                                 strvcat('Title Top Right','Title Bottom Right')}}
%                          Careful: note transpose structure
%                     Can create a single row of titles accross the top with:
%                               {{'title1' 'title2'}}
%                     To skip titles on some plots, put a space string ' '
%                     in those locations
%
%    titlefontsize    font size for title
%                           (only has effect when plot is made here,
%                            and when the titlecellstr is nonempty)
%                     default is empty, [], for Matlab default
%
%    labelcellstr    Vertical cell array of strings for axis labels
%                        create this using cellstr, 
%                        or {{string1; string2; ...}}
%                            Note:  These strange double brackets seems to be 
%                                needed for correct pass to subroutine
%                                It may change in later versions of Matlab
%                    Default is an empty cell array, {}, which then gives:
%                         For iMDdir = 0, data dimension d = <= 4 plot:
%                            "Direction _" for each
%                         For iMDdir = 0, data dimension d = > 4 plot:
%                            "PC _" for each
%                         For iMDdir = 1:
%                            "Mean Diff Direction" 1st
%                            "Ortho PC _" for others
%                    For no labels, use {''; ''; ''; ''}
%                    Length of cell array must be:
%                        1 for data dimension d = 1
%                        2 for data dimension d = 2
%                        3 for data dimension d = 3
%                        4 for data dimension d = >= 4
%
%    labelfontsize    font size for axis labels
%                                    (only has effect when plot is made here,
%                                     and when a label str is nonempty)
%                           default is empty [], for Matlab default
%
%
%    savestr          string controlling saving of output,
%                         either a full path, or a file prefix to
%                         save in matlab's current directory
%                         Will add .ps, and save as either
%                             color postscript (for plot with Entry 3 = 1)
%                         or
%                             black&white postscript (other plots)
%                         unspecified:  results only appear on screen
%
%    iscreenwrite     0  (default)  no screen writes
%                     1  write to screen to show progress
%
%
% Outputs:
%     Graphics in current Figure
%     When savestr exists,
%        Postscript file saved in 'savestr'.ps
%
% Assumes path can find personal functions:
%    scatplotSM.m
%    bwsjpiSM.m
%    kdeSM.m
%    lbinrSM.m
%    vec2matSM.m
%    pcaSM.m
%    projplot1SM.m
%    projplot2SM.m
%    bwrfphSM.m
%    bwosSM.m
%    rootfSM
%    bwrotSM.m
%    bwsnrSM.m
%    iqrSM.m
%    cquantSM.m
%    axisSM.m

%    Copyright (c) J. S. Marron 2007



%  First set all parameters to defaults
%
iMDdir = 1 ;
icolor = [[1 0 0]; [0 0 1]] ;
markerstr = strvcat('o','+') ;
legendcellstr = {} ;
mlegendcolor = [] ;
maxlim = [] ;
iplotaxes = 1 ;
iplotdirvec = 0 ;
ibelowdiag = 1 ;
titlecellstr = {} ;
titlefontsize = [] ;
labelcellstr = {} ;
labelfontsize = [] ;
savestr = [] ;
iscreenwrite = 0 ;


%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 2 ;   %  then paramstruct is an argument

  if isfield(paramstruct,'iMDdir') ;    %  then change to input value
    iMDdir = getfield(paramstruct,'iMDdir') ; 
  end ;

  if isfield(paramstruct,'icolor') ;    %  then change to input value
    icolor = getfield(paramstruct,'icolor') ; 
  end ;

  if isfield(paramstruct,'markerstr') ;    %  then change to input value
    markerstr = getfield(paramstruct,'markerstr') ; 
  end ;

  if isfield(paramstruct,'legendcellstr') ;    %  then change to input value
    legendcellstr = getfield(paramstruct,'legendcellstr') ; 
  end ;

  if isfield(paramstruct,'mlegendcolor') ;    %  then change to input value
    mlegendcolor = getfield(paramstruct,'mlegendcolor') ; 
  end ;

  if isfield(paramstruct,'maxlim') ;    %  then change to input value
    maxlim = getfield(paramstruct,'maxlim') ; 
  end ;

  if isfield(paramstruct,'iplotaxes') ;    %  then change to input value
    iplotaxes = getfield(paramstruct,'iplotaxes') ; 
  end ;

  if isfield(paramstruct,'iplotdirvec') ;    %  then change to input value
    iplotdirvec = getfield(paramstruct,'iplotdirvec') ; 
  end ;

  if isfield(paramstruct,'ibelowdiag') ;    %  then change to input value
    ibelowdiag = getfield(paramstruct,'ibelowdiag') ; 
  end ;

  if isfield(paramstruct,'titlecellstr') ;    %  then change to input value
    titlecellstr = getfield(paramstruct,'titlecellstr') ; 
  end ;

  if isfield(paramstruct,'titlefontsize') ;    %  then change to input value
    titlefontsize = getfield(paramstruct,'titlefontsize') ; 
  end ;

  if isfield(paramstruct,'labelcellstr') ;    %  then change to input value
    labelcellstr = getfield(paramstruct,'labelcellstr') ; 
  end ;

  if isfield(paramstruct,'labelfontsize') ;    %  then change to input value
    labelfontsize = getfield(paramstruct,'labelfontsize') ; 
  end ;

  if isfield(paramstruct,'savestr') ;    %  then use input value
    savestr = getfield(paramstruct,'savestr') ; 
    if ~(ischar(savestr) | isempty(savestr)) ;    %  then invalid input, so give warning
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from SigClustLabelPlotSM:  !!!') ;
      disp('!!!   Invalid savestr,                   !!!') ;
      disp('!!!   using default of no save           !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      savestr = [] ;
    end ;
  end ;

  if isfield(paramstruct,'iscreenwrite') ;    %  then change to input value
    iscreenwrite = getfield(paramstruct,'iscreenwrite') ; 
  end ;

end ;    %  of resetting of input parameters



%  Check Validity of Inputs
%      (Note that most checking is done in scatplotSM.m)
%
d = size(data,1) ;
         %  dimension of each data curve
n = size(data,2) ;
         %  number of data curves
vflag1 = (vclass == 1) ;
vflag2 = (vclass == 2) ;

if (size(vclass,1) ~= 1) | (size(vclass,2) ~= n) | ...
       (sum(vflag1) + sum(vflag2) ~= n) ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Warning from SigClustLabelPlotSM.m:   !!!') ;
  disp('!!!   Invalid vclass,                       !!!') ;
  disp('!!!   Size must be 1xn                      !!!') ;
  disp('!!!   Must contain only 1s and 2s           !!!') ;
  disp('!!!   Terminating Execution                 !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;
end ;

if (size(icolor,1) ~= 2) | (size(icolor,2) ~= 3) ;
  if ~((size(icolor,1) == 1) & (size(icolor,2) == 1) & (icolor == 0)) ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from SigClustLabelPlotSM.m:   !!!') ;
    disp('!!!   Invalid size of icolor,               !!!') ;
    disp('!!!   Must be 2x3                           !!!') ;
    disp('!!!   Resetting to all black                !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    icolor = 0 ;
  end ;
end ;

if (size(markerstr,1) ~= 2) | (size(markerstr,2) ~= 1) | (~ischar(markerstr)) ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Warning from SigClustLabelPlotSM.m:   !!!') ;
  disp('!!!   Invalid markerstr,                    !!!') ;
  disp('!!!   Size must be 2x1                      !!!') ;
  disp('!!!   Must be a character array             !!!') ;
  disp('!!!   Resetting to default                  !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  markerstr = strvcat('o','+') ;
end ;

if (length(legendcellstr) == 2) & isempty(mlegendcolor) ;
  mlegendcolor = icolor ;
end ;



if d == 1 ;    %  then use projplot1SM.m 

  %  Set paramstruct for call to scatplotSM.m
  %
  if icolor == 0 ;
    icolorin = 0 ;
    mlegendcolor = [] ;
  else ;
    icolorin = vflag1' * icolor(1,:) + vflag2' * icolor(2,:) ;
  end ;

  markerstrin = [] ;
  for i = 1:n ;
    if vflag1(i) ;
      markerstrin = strvcat(markerstrin,markerstr(1,:)) ;
    else ;
      markerstrin = strvcat(markerstrin,markerstr(2,:)) ;
    end ;
  end ;

  paramstruct = struct('icolor',icolorin, ...
                       'markerstr',markerstrin, ...
                       'vaxlim',maxlim, ...
                       'titlefontsize',titlefontsize, ...
                       'labelfontsize',labelfontsize, ...
                       'savestr',savestr, ...
                       'iscreenwrite',iscreenwrite) ;

  %  Update to properly handle empty components
  %
  if ~(isempty(legendcellstr)) ;
    paramstruct = setfield(paramstruct,'legendcellstr',legendcellstr) ;
  end ;
  if ~(isempty(mlegendcolor)) ;
    paramstruct = setfield(paramstruct,'mlegendcolor',mlegendcolor) ;
  end ;
  if ~(isempty(titlecellstr)) ;
    paramstruct = setfield(paramstruct,'titlestr',titlecellstr{1}) ;
  end ;
  if ~(isempty(labelcellstr)) ;
    paramstruct = setfield(paramstruct,'labelstr',labelcellstr{1}) ;
  end ;

  projplot1SM(data,1,paramstruct) ;


else ;    %  then use scatplotSM.m

  %  Set directions
  %
  if iMDdir == 0 ;    %  Then use ordinary directions
                      %      (regular for d = 2-4, PC otherwise)

    if d == 1 ;
      mdir = 1 ;
      npcadiradd = 0 ;
    elseif  d == 2  |  d == 3  |  d == 4  ;
      mdir = eye(d) ;
      npcadiradd = 0 ;
    else ;
      mdir = [] ;
      npcadiradd = 4 ;
    end ;

  else ;    %  Then use Mean Difference and orthogonal PC directions

    if d == 1 ;
      mdir = 1 ;
      npcadiradd = 0 ;
    elseif  d == 2  |  d == 3  ;
      vMD = mean(data(:,vflag1),2) - mean(data(:,vflag2),2) ;
      vMD = vMD / norm(vMD) ;
      mdir = vMD ;
      npcadiradd = -(d - 1) ;
    else ;
      vMD = mean(data(:,vflag1),2) - mean(data(:,vflag2),2) ;
      vMD = vMD / norm(vMD) ;
      mdir = vMD ;
      npcadiradd = -3 ;
    end ;

  end ;



  %  Set paramstruct for call to scatplotSM.m
  %
  if icolor == 0 ;
    icolorin = 0 ;
    mlegendcolor = [] ;
  else ;
    icolorin = vflag1' * icolor(1,:) + vflag2' * icolor(2,:) ;
  end ;

  markerstrin = [] ;
  for i = 1:n ;
    if vflag1(i) ;
      markerstrin = strvcat(markerstrin,markerstr(1,:)) ;
    else ;
      markerstrin = strvcat(markerstrin,markerstr(2,:)) ;
    end ;
  end ;

  paramstruct = struct('icolor',icolorin, ...
                       'npcadiradd',npcadiradd, ...
                       'markerstr',markerstrin, ...
                       'maxlim',maxlim, ...
                       'iplotaxes',iplotaxes, ...
                       'iplotdirvec',iplotdirvec, ...
                       'ibelowdiag',ibelowdiag, ...
                       'titlefontsize',titlefontsize, ...
                       'labelfontsize',labelfontsize, ...
                       'savestr',savestr, ...
                       'iscreenwrite',iscreenwrite) ;

  %  Update to properly handle empty components
  %
  if ~(isempty(legendcellstr)) ;
    paramstruct = setfield(paramstruct,'legendcellstr',legendcellstr) ;
  end ;
  if ~(isempty(mlegendcolor)) ;
    paramstruct = setfield(paramstruct,'mlegendcolor',mlegendcolor) ;
  end ;
  if ~(isempty(titlecellstr)) ;
    paramstruct = setfield(paramstruct,'titlecellstr',titlecellstr) ;
  end ;
  if ~(isempty(labelcellstr)) ;
    paramstruct = setfield(paramstruct,'labelcellstr',labelcellstr) ;
  elseif iMDdir ~= 0 ;
    paramstruct = setfield(paramstruct,'labelcellstr',{'Mean Diff Direction'}) ;
  end ;

  scatplotSM(data,mdir,paramstruct) ;


end ;




