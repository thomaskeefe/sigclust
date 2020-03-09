function  [BestClass, bestCI] = SigClust2meanFastSM(data,paramstruct) 
% SIGCLUST2MEANFASTSM, statistical SIGnificance of CLUSTers,
%         computes 2-MEAN clustering using a FAST algorithm
%   Steve Marron's matlab function
%     Provides a fast (relative to many random restarts)
%     algorithm for the 2-means clustering
%     method for splitting a given data set.
%     The main goal is to find the global minimizer, 
%     without requiring the usual large number of 
%     random restarts.  Idea is to use PCA to find 
%     suggested starting classifications for the usual
%     algorithm, and to work in appropriate orthogonal 
%     subspaces for later steps.
%     Results are not guaranteed, but are expected to be
%     of high quality for many High Dimension Low Sample
%     Size data sets, because of the theoretically
%     predicted structure of such data 
%     (e.g. relative orthogonality of clusters)
%
% Inputs:
%   data    - d x n matrix of data, each column vector is 
%                    a "d-dim data vector"
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
%    maxstep          MAXimum number of STEPs to consider.
%                     When not specified, default is 10
%
%    ioutplot         0  make no output plots
%                     1 (default) make output plots at end of each step
%                           in new coorinate system
%                     2  make output plots at end of each step, 
%                           in both old and new coordinate systems
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
%    maxlim        Matrix of axis limits
%                        Use [] for default of all automatically chosen, 
%                            by axisSM
%                        Use 1 for symmetrically chosen, by axisSM
%                            (often preferred for centered plots, as in PCA)
%                        Otherwise, must be (size(mdir,2) + npcadir) x 2 
%                            matrix of axis limits, 
%                            with each row corresponding to a direction
%                        Note:  Use is generally not recommended,
%                        because defaults give "good visual impression
%                        of decomposition".  It is mostly intended to allow
%                        the highlighting of "visually different scales" 
%                        in data.  But this comes at the cost of reduced  
%                         detail being visible in the plots
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
%    titlestr         string to add to subplot titles
%                     always get the left top title indicating Step #
%                     the second top title indicating view
%                     this controls the third top title
%                     default is an empty array, [] for no additional title
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
%                         For istep Starting Cluster plot:
%                            "Mean Diff Direction" 1st
%                            "Ortho PC _" for others
%                         For istep 2-Means Result, data dimension d = <= 4 plot:
%                            "Direction _" for each
%                         For istep 2-Means Result, data dimension d = > 4 plot:
%                            "PC _" for each
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
% Outputs:
%    BestClass        Best Cluster Labelling  
%                     (in sense of minimum Cluster Index, over repetitions)
%                     1 x n vector of labelings, i.e. 1's and 2's  
%
%    bestCI           Best Cluster Index value
%                     scalar
%
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
maxstep = 10 ;
ioutplot = 1 ;
icolor = [[1 0 0]; [0 0 1]] ;
markerstr = strvcat('o','+') ;
maxlim = [] ;
iplotaxes = 1 ;
iplotdirvec = 0 ;
ibelowdiag = 1 ;
titlestr = [] ;
titlefontsize = [] ;
labelcellstr = {} ;
labelfontsize = [] ;
savestr = [] ;
iscreenwrite = 0 ;


%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 1 ;   %  then paramstruct is an argument

  if isfield(paramstruct,'maxstep') ;    %  then change to input value
    maxstep = getfield(paramstruct,'maxstep') ; 
  end ;

  if isfield(paramstruct,'ioutplot') ;    %  then change to input value
    ioutplot = getfield(paramstruct,'ioutplot') ; 
  end ;

  if isfield(paramstruct,'icolor') ;    %  then change to input value
    icolor = getfield(paramstruct,'icolor') ; 
  end ;

  if isfield(paramstruct,'markerstr') ;    %  then change to input value
    markerstr = getfield(paramstruct,'markerstr') ; 
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

  if isfield(paramstruct,'titlestr') ;    %  then change to input value
    titlestr = getfield(paramstruct,'titlestr') ; 
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
      disp('!!!   Warning from SigClust2meanFastSM:  !!!') ;
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



if ioutplot ~= 0 ;  %  Will make plots, so close current windows
  close all ;
end ;

rankdata = rank(data) ;


%  Start with PC1 labelling
%
if size(data,1) > 1 ;    %  d > 1, so actually do PCA

  viout = [0 0 0 0 1] ;    %  Output mpc  -  npc x n matrix of 
                           %     "principal components", 
                           %     i.e. of coefficients of projections 
                           %     of data onto eigenvectors, 
                           %     i.e. of "scores".
  paramstruct = struct('npc',1,...
                       'viout',viout,...
                       'iscreenwrite',0) ;
  outstruct = pcaSM(data,paramstruct) ;
  vpc = getfield(outstruct,'mpc') ;
else ;
  vpc = data ;
end ;
curClass = (vpc > median(vpc)) ;
    %  vector of 0s and 1s for class labels
curClass = curClass + 1 ;
    %  vector of 1s and 2s for class labels
curCI = ClustIndSM(data,(curClass == 1),(curClass == 2)) ;
if iscreenwrite == 1 ;
  disp(['    2 Means Fast, PC1 Cluster Index = ' num2str(curCI)]) ;
end ;

if ioutplot == 2 ;  %  Make output plot using 
                    %    original data PCA coordinates

  figure ;
  if isempty(titlestr) ;
    titlecellstr = {{['2-Means Fast, Step ' num2str(1)] 'Starting Cluster'}} ;
  else ;
    titlecellstr = {{['2-Means Fast, Step ' num2str(1)] 'Starting Cluster' titlestr}} ;
  end ;
  if ~isempty(savestr) ;
    savestrout = [savestr 'Step' num2str(1) 'StartClust'] ;
  else ;
    savestrout = [] ;
  end ;
  paramstruct = struct('iMDdir',0,...
                       'icolor',icolor,...
                       'markerstr',markerstr,...
                       'legendcellstr',{{' ' ['CI = ' num2str(curCI)]}},...
                       'mlegendcolor',zeros(2,3),...
                       'maxlim',maxlim,...
                       'iplotaxes',iplotaxes,...
                       'iplotdirvec',iplotdirvec,...
                       'ibelowdiag',ibelowdiag,...
                       'titlecellstr',titlecellstr,...
                       'titlefontsize',titlefontsize,...
                       'labelfontsize',labelfontsize,...
                       'savestr',savestrout,...
                       'iscreenwrite',0) ;

  %  Update to properly handle empty components
  %
  if ~(isempty(labelcellstr)) ;
    paramstruct = setfield(paramstruct,'labelcellstr',labelcellstr{1}) ;
  end ;

  SigClustLabelPlotSM(data,curClass,paramstruct) ; 

end ;



%  Compute 1st 2 means clustering
%
CurMMeanStart = [mean(data(:,(curClass == 1)),2) mean(data(:,(curClass == 2)),2)] ;
[idx,mc] = kmeans(data',2,'EmptyAction','singleton','Start',CurMMeanStart') ;
curClass = idx' ;
curCI = ClustIndSM(data,(curClass == 1),(curClass == 2)) ;
CurMMeanStart = mc' ;
if iscreenwrite == 1 ;
  disp(['    2 Means Fast, Step 1 Cluster Index = ' num2str(curCI)]) ;
end ;
MDdir = CurMMeanStart(:,1) - CurMMeanStart(:,2) ;
MDdir = MDdir / norm(MDdir) ;
mdirexc = MDdir ;
    %  matrix of directions to exclude

if ioutplot ~= 0 ;  %  Make output plot using 
                    %    new PCA coordinates


  figure ;
  if isempty(titlestr) ;
    titlecellstr = {{['2-Means Fast, Step ' num2str(1)] '2-means Result'}} ;
  else ;
    titlecellstr = {{['2-Means Fast, Step ' num2str(1)] '2-means Result' titlestr}} ;
  end ;
  if ~isempty(savestr) ;
    savestrout = [savestr 'Step' num2str(1) 'Res2mean'] ;
  else ;
    savestrout = [] ;
  end ;
  paramstruct = struct('iMDdir',1,...
                       'icolor',icolor,...
                       'markerstr',markerstr,...
                       'legendcellstr',{{' ' ['CI = ' num2str(curCI)]}},...
                       'mlegendcolor',zeros(2,3),...
                       'maxlim',maxlim,...
                       'iplotaxes',iplotaxes,...
                       'iplotdirvec',iplotdirvec,...
                       'ibelowdiag',ibelowdiag,...
                       'titlecellstr',titlecellstr,...
                       'titlefontsize',titlefontsize,...
                       'labelfontsize',labelfontsize,...
                       'savestr',savestrout,...
                       'iscreenwrite',0) ;

  %  Update to properly handle empty components
  %
  if ~(isempty(labelcellstr)) ;
    paramstruct = setfield(paramstruct,'labelcellstr',labelcellstr{1}) ;
  end ;

  SigClustLabelPlotSM(data,curClass,paramstruct) ; 

end ;


bestCI = curCI ;
BestClass = curClass ;



for istep = 2:maxstep ;

  if size(mdirexc,2) < rankdata ;    %  then can work with non-excluded data

    %  Project data on non-excluded directions
    %
    mprojexc = mdirexc * pinv(mdirexc' * mdirexc) * mdirexc' ;
        %  projection matrix onto excluded space
    projdata = data - mprojexc * data ;


    %  PC1 labelling on nonexcluded data
    %
    viout = [0 1 0 0 1] ;    %  Output mpc  -  npc x n matrix of 
                             %     "principal components", 
                             %     i.e. of coefficients of projections 
                             %     of data onto eigenvectors, 
                             %     i.e. of "scores".
    paramstruct = struct('npc',1,...
                         'viout',viout,...
                         'iscreenwrite',0) ;
    outstruct = pcaSM(projdata,paramstruct) ;
    vpc = getfield(outstruct,'mpc') ;
    vcurPCdir = getfield(outstruct,'meigvec') ;
    curClass = (vpc > median(vpc)) ;
        %  vector of 0s and 1s for class labels
    curClass = curClass + 1 ;
        %  vector of 1s and 2s for class labels
    curCI = ClustIndSM(data,(curClass == 1),(curClass == 2)) ;
    if curCI < bestCI ;
      bestCI = curCI ;
      BestClass = curClass ;
    end ;
    if iscreenwrite == 1 ;
      disp(['    2 Means Fast, Step ' num2str(istep) ' Ortho PC1 Cluster Index = ' num2str(curCI)]) ;
    end ;

    if ioutplot == 2 ;  %  Make output plot using 
                        %    original data PCA coordinates

      figure ;
      if isempty(titlestr) ;
        titlecellstr = {{['2-Means Fast, Step ' num2str(istep)] 'Starting Cluster'}} ;
      else ;
        titlecellstr = {{['2-Means Fast, Step ' num2str(istep)] 'Starting Cluster' titlestr}} ;
      end ;
      if ~isempty(savestr) ;
        savestrout = [savestr 'Step' num2str(istep) 'StartClust'] ;
      else ;
        savestrout = [] ;
      end ;
      paramstruct = struct('iMDdir',0,...
                           'icolor',icolor,...
                           'markerstr',markerstr,...
                           'legendcellstr',{{' ' ['CI = ' num2str(curCI)]}},...
                           'mlegendcolor',zeros(2,3),...
                           'maxlim',maxlim,...
                           'iplotaxes',iplotaxes,...
                           'iplotdirvec',iplotdirvec,...
                           'ibelowdiag',ibelowdiag,...
                           'titlecellstr',titlecellstr,...
                           'titlefontsize',titlefontsize,...
                           'labelfontsize',labelfontsize,...
                           'savestr',savestrout,...
                           'iscreenwrite',0) ;

      %  Update to properly handle empty components
      %
      if ~(isempty(labelcellstr)) ;
        paramstruct = setfield(paramstruct,'labelcellstr',labelcellstr{1}) ;
      end ;

      SigClustLabelPlotSM(data,curClass,paramstruct) ; 

    end ;



    %  Compute next 2 means clustering
    %
    CurMMeanStart = [mean(data(:,(curClass == 1)),2) mean(data(:,(curClass == 2)),2)] ;
    [idx,mc] = kmeans(data',2,'EmptyAction','singleton','Start',CurMMeanStart') ;
    curClass = idx' ;
    curCI = ClustIndSM(data,(curClass == 1),(curClass == 2)) ;
    CurMMeanStart = mc' ;
    if curCI < bestCI ;
      bestCI = curCI ;
      BestClass = curClass ;
    end ;
    if iscreenwrite == 1 ;
      disp(['    2 Means Fast, Step ' num2str(istep) ' Cluster Index = ' num2str(curCI)]) ;
    end ;
    MDdir = CurMMeanStart(:,1) - CurMMeanStart(:,2) ;
    MDdir = MDdir / norm(MDdir) ;

    if sum(abs(mprojexc * MDdir - MDdir)) > 0 ;
                            %  this direction already in excluded subspace
      mdirexc = [mdirexc vcurPCdir] ;
          %  matrix of directions to exclude
    else ;    %  have component outside current subspace
      mdirexc = [mdirexc MDdir] ;
          %  matrix of directions to exclude
    end ;

    if ioutplot ~= 0 ;  %  Make output plot using 
                        %    new PCA coordinates

      figure ;
      if isempty(titlestr) ;
        titlecellstr = {{['2-Means Fast, Step ' num2str(istep)] '2-means Result'}} ;
      else ;
        titlecellstr = {{['2-Means Fast, Step ' num2str(istep)] '2-means Result' titlestr}} ;
      end ;
      if ~isempty(savestr) ;
        savestrout = [savestr 'Step' num2str(istep) 'Res2mean'] ;
      else ;
        savestrout = [] ;
      end ;
      paramstruct = struct('iMDdir',1,...
                           'icolor',icolor,...
                           'markerstr',markerstr,...
                           'legendcellstr',{{' ' ['CI = ' num2str(curCI)]}},...
                           'mlegendcolor',zeros(2,3),...
                           'maxlim',maxlim,...
                           'iplotaxes',iplotaxes,...
                           'iplotdirvec',iplotdirvec,...
                           'ibelowdiag',ibelowdiag,...
                           'titlecellstr',titlecellstr,...
                           'titlefontsize',titlefontsize,...
                           'labelfontsize',labelfontsize,...
                           'savestr',savestrout,...
                           'iscreenwrite',0) ;

      %  Update to properly handle empty components
      %
      if ~(isempty(labelcellstr)) ;
        paramstruct = setfield(paramstruct,'labelcellstr',labelcellstr{1}) ;
      end ;

      SigClustLabelPlotSM(data,curClass,paramstruct) ; 

    end ;


  end ;


end ;



