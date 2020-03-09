function [BestClass, bestCI] = SigClust2meanPerfSM(data,paramstruct) 
% SIGCLUST2MEANPERFSM, statistical SIGnificance of CLUSTers,
%         checks 2-MEAN clustering PERFormance, for given data 
%   Steve Marron's matlab function
%     Studies how consistently the standard 2-means clustering
%     method splits a given data set.
%     Does a large number of repetitions, using random 
%     starts, and summarizes answers.
%     Allows image plot visualizations.
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
%    nrep             Number of repetitions (random restarts).
%                     When not specified, default is 100
%
%    viplot           vector of indices 0 or 1 for various types of output plots
%                                   1 in a given entry will include that 
%                                   in a new Figure Window
%                     Entry 1:  Make plot of Cluster Index vs. Restart Number
%                                   both sorted and unsorted.
%                     Entry 2:  Image plot, Showing Cluster Results
%                                   Random Start Number vs. Case Number
%                     Entry 3:  Image plot, Showing Cluster Results
%                                   Sorted (on Cluster Index) Start Number 
%                                   vs. Case Number
%                     Entry 4:  Image plot, Showing Cluster Results
%                                   Sorted Start Number vs. Case Number
%                                   Highlight most common restarts
%                     Entry 5:  Image plot, Showing Cluster Results
%                                   Sorted Start Number vs. Case Number
%                                   Highlight those similar to most common 
%                     Entry 6:  Image plot, Showing Cluster Results
%                                   Sorted Start Number vs. Case Number
%                                   Restarts Flipped, to be similar to most common
%                     Entry 7:  Image plot, Showing Cluster Results
%                                   Flipped Restarts
%                                   Sorted Start Number vs. Sorted Case Number
%                     When not specified, default is:  [1 1 1 1 1 1 1] 
%
%    randstate        State of uniform random number generator
%                     When empty, or not specified, just use current seed  
%
%    randnstate       State of normal random number generator
%                     When empty, or not specified, just use current seed  
%
%    titlestr         string to start titles of output plot
%                     (generally should end in ', ', of perhaps ',     ') 
%
%    titlefontsize    font size for title
%                                    (only has effect when plot is made here,
%                                     and when the titlestr is nonempty)
%                           default is empty [], for Matlab default,
%                           (18 seems to look good in printed .ps file)
%
%    labelfontsize    font size for axis labels
%                                    (only has effect when plot is made here)
%                           default is empty [], for Matlab default
%                           (12 seems to look good in printed .ps file)
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
%        Postscript files saved in 'savestr'.ps
%
%     BestClass        Best Cluster Labelling  
%                     (in sense of minimum Cluster Index, over repetitions)
%                     1 x n vector of labelings, i.e. 1's and 2's  
%
%     bestCI           Best Cluster Index value
%                     scalar
%
% Assumes path can find personal functions:
%    vec2matSM.m
%    axisSM.m

%    Copyright (c) J. S. Marron 2007



%  First set all parameters to defaults
%
nrep = 100 ;
viplot = [1 1 1 1 1 1 1] ;
randstate = [] ;
randnstate = [] ;
titlestr = [] ;
titlefontsize = [] ;
labelfontsize = [] ;
savestr = [] ;
iscreenwrite = 0 ;


%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 1 ;   %  then paramstruct is an argument

  if isfield(paramstruct,'nrep') ;    %  then change to input value
    nrep = getfield(paramstruct,'nrep') ; 
  end ;

  if isfield(paramstruct,'viplot') ;    %  then change to input value
    viplot = getfield(paramstruct,'viplot') ; 
  end ;

  if isfield(paramstruct,'randstate') ;    %  then change to input value
    randstate = getfield(paramstruct,'randstate') ; 
  end ;

  if isfield(paramstruct,'randnstate') ;    %  then change to input value
    randnstate = getfield(paramstruct,'randnstate') ; 
  end ;

  if isfield(paramstruct,'titlestr') ;    %  then change to input value
    titlestr = getfield(paramstruct,'titlestr') ; 
  end ;

  if isfield(paramstruct,'titlefontsize') ;    %  then change to input value
    titlefontsize = getfield(paramstruct,'titlefontsize') ; 
  end ;

  if isfield(paramstruct,'labelfontsize') ;    %  then change to input value
    labelfontsize = getfield(paramstruct,'labelfontsize') ; 
  end ;

  if isfield(paramstruct,'savestr') ;    %  then use input value
    savestr = getfield(paramstruct,'savestr') ; 
    if ~(ischar(savestr) | isempty(savestr)) ;    %  then invalid input, so give warning
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from SigClust2meanPerfSM:  !!!') ;
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



%  Readjust viplot as needed
%
maxviplot = 7 ;
    %  largest useful size for viout 
if size(viplot,1) > 1 ;    %  if have more than one row
  viplot = viplot' ;
end ;
if size(viplot,1) == 1 ;    %  then have row vector

  if length(viplot) < maxviplot ;    %  then pad with 0s
    viplot = [viplot zeros(1,maxviplot - length(viplot))] ;
  end ;

else ;    %  invalid viplot, so indicate and quit

  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from SigClust2meanPerfSM:  !!!') ;
  disp('!!!   Invalid viplot                   !!!') ;
  disp('!!!   Terminating exceution            !!!') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  return ;

end ;



%  set preliminary stuff
%
d = size(data,1) ;
         %  dimension of each data curve
n = size(data,2) ;
         %  number of data curves



%  Run nrep 2-means clusterings, with random restarts
%
paramstruct = struct('nrep',nrep,...
                     'randstate',randstate,...
                     'randnstate',randnstate,...
                     'iscreenwrite',iscreenwrite) ;
[BestClass, vindex, midx] = SigClust2meanRepSM(data,paramstruct) ;



nfigopen = 0 ;
if viplot(1) == 1 ;    %  Make plot of Cluster Index vs. Restart Number
                       %            both sorted and unsorted.

  nfigopen = nfigopen + 1 ;
  figure(nfigopen) ;
  clf ;

  [vindexsort,vindsort] = sort(vindex) ;

  subplot(2,1,1) ;
    plot(1:nrep,vindex,'ko') ;
      vax = axisSM(1:nrep,vindex) ;
      if vax(3) == vax(4) ;
        vax(3) = 0 ;
        vax(4) = 1 ;
      end ;
      axis([0, (nrep + 1), vax(3), vax(4)]) ;
      if isempty(titlefontsize) ;
        title([titlestr '2 - Means CI, over ' num2str(nrep) ' random restarts']) ;
      else ;
        title([titlestr '2 - Means CI, over ' num2str(nrep) ' random restarts'], ...
              'FontSize',titlefontsize) ;
      end ;
      if isempty(labelfontsize) ;
        xlabel('Random Start Number') ;
        ylabel('Cluster Index') ;
      else ;
        xlabel('Random Start Number','FontSize',labelfontsize) ;
        ylabel('Cluster Index','FontSize',labelfontsize) ;
      end ;
      text(0.1 * (nrep + 1), ...
           vax(3) + 0.9 * (vax(4) - vax(3)), ...
           ['Proportional Difference = ' num2str((vindexsort(nrep) - vindexsort(1)) / vindexsort(1))]) ;

  subplot(2,1,2) ;
    plot(1:nrep,vindexsort,'ko') ;
      axis([0, (nrep + 1), vax(3), vax(4)]) ;
      if isempty(titlefontsize) ;
        title([titlestr 'Sorted Version of Above']) ;
      else ;
        title([titlestr 'Sorted Version of Above'], ...
              'FontSize',titlefontsize) ;
      end ;
      if isempty(labelfontsize) ;
        xlabel('Sorted (on Index) Start Number') ;
        ylabel('Cluster Index') ;
      else ;
        xlabel('Sorted (on Index) Start Number','FontSize',labelfontsize) ;
        ylabel('Cluster Index','FontSize',labelfontsize) ;
      end ;

  if ~isempty(savestr) ;
    orient landscape ;
    print('-dps2',[savestr 'Clust2Index.ps']) ;
  end ;


end ;



if viplot(2) == 1 ;    %  Image plot, Showing Cluster Results
                           %        Random Start Number vs. Case Number

  nfigopen = nfigopen + 1 ;
  figure(nfigopen) ;
  clf ;

  colormap([ones(1,3); zeros(1,3)]) ;
  image(midx) ;
    if isempty(titlefontsize) ;
      title([titlestr 'Image of Cluster Labels']) ;
    else ;
      title([titlestr 'Image of Cluster Labels'], ...
            'FontSize',titlefontsize) ;
    end ;
    if isempty(labelfontsize) ;
      xlabel('Case Number') ;
      ylabel('Random Start Number') ;
    else ;
      xlabel('Case Number','FontSize',labelfontsize) ;
      ylabel('Random Start Number','FontSize',labelfontsize) ;
    end ;

  if ~isempty(savestr) ;
    orient landscape ;
    print('-dps2',[savestr 'ImageClustRaw.ps']) ;
  end ;


end ;



if sum(viplot(3:end) > 0.5) ;    %  then need to sort rows by cluster index  

  [vindexsort,vindsort] = sort(vindex) ;
  midxs = midx(vindsort,:) ;
      %  Sort rows by cluster index


  if viplot(3) == 1 ;    %  Image plot, Showing Cluster Results
                         %        Sorted (on Cluster Index Start Number vs. Case Number

    nfigopen = nfigopen + 1 ;
    figure(nfigopen) ;
    clf ;

    colormap([ones(1,3); zeros(1,3)]) ;
    image(midxs) ;
      if isempty(titlefontsize) ;
        title([titlestr 'Sort Rows on Cluster Index']) ;
      else ;
        title([titlestr 'Sort Rows on Cluster Index'], ...
              'FontSize',titlefontsize) ;
      end ;
      if isempty(labelfontsize) ;
        xlabel('Case Number') ;
        ylabel('Sorted (on Cluster Index) Start Number') ;
      else ;
        xlabel('Case Number','FontSize',labelfontsize) ;
        ylabel('Sorted (on Cluster Index) Start Number','FontSize',labelfontsize) ;
      end ;

    if ~isempty(savestr) ;
      orient landscape ;
      print('-dps2',[savestr 'ImageClustSortR.ps']) ;
    end ;


  end ;



  if sum(viplot(4:end) > 0.5) ;    %  then need to find most common restarts

    vncommonindex = [] ;
    for irep = 1:nrep ;    %  loop through restarts
      vidxs = midxs(irep,:) ;
          %  vector of cluster indices, for this restart
      ncommonindex = 0 ;
          %  will count number of restarts
          %  with same clustering
      for irepp = 1:nrep ;
        if sum(vidxs == midxs(irepp,:)) == n ;    
                        % has same clustering
          ncommonindex = ncommonindex + 1 ;
        end ;
      end ;
      vncommonindex = [vncommonindex; ncommonindex] ;
    end ;

    [temp,imax] = max(vncommonindex) ;
        %  index of maximizer of vncommonindex

    vidxcommon = midxs(imax,:) ;
        %  vector of indices for most common restart
    vcommonset = [] ;
    for irep = 1:nrep ;    %  loop through restarts
      if sum(vidxcommon == midxs(irep,:)) == n ;    
                      % are always assigned to the same cluster
        vcommonset = [vcommonset; irep] ;
            %  then keep this index among common set
      end ;
    end ; 



    if viplot(4) == 1 ;    %  Image plot, Showing Cluster Results
                           %        Sorted Start Number vs. Case Number
                           %        Highlight most common restarts

      nfigopen = nfigopen + 1 ;
      figure(nfigopen) ;
      clf ;

      midxsc = midxs ;
      midxsc(vcommonset,:) = midxsc(vcommonset,:) + 2 * ones(length(vcommonset),n) ;
           %  upshift by 2 the members of the common set

      colormap([ones(1,3); zeros(1,3); [1 0 0 ]; [0 0 1]]) ;
      image(midxsc) ;
        if isempty(titlefontsize) ;
          title([titlestr 'Most Common Colored']) ;
        else ;
          title([titlestr 'Most Common Colored'], ...
                'FontSize',titlefontsize) ;
        end ;
        if isempty(labelfontsize) ;
          xlabel('Case Number') ;
          ylabel('Sorted (on Cluster Index) Start Number') ;
        else ;
          xlabel('Case Number','FontSize',labelfontsize) ;
          ylabel('Sorted (on Cluster Index) Start Number','FontSize',labelfontsize) ;
        end ;

      if ~isempty(savestr) ;
        orient landscape ;
        print('-dpsc2',[savestr 'ImageClustComCol.ps']) ;
      end ;


    end ;



    if sum(viplot(5:end) > 0.5) ;    %  then need to find those similar to most common restarts

      visimilar = [] ;
          %  vector if indices of restarts that similar 
          %  in at least half the reprtitions
      for irep = 1:nrep ;    %  loop through cases
        countsim = sum(vidxcommon == midxs(irep,:)) ;
        if countsim > n / 2 ;    
                        % are assigned to the same cluster more than half the time
          visimilar = [visimilar; irep] ;
              %  then keep this index among similar set
        elseif countsim == n / 2 ;    
                        % are assigned to the same cluster exactly half the time
          if midxs(irep,1) == vidxcommon(1) ;
                          %  break tie, by using 1st case
            visimilar = [visimilar; irep] ;
                %  then keep this index among similar set
          end ;
        end ;
      end ; 


      if viplot(5) == 1 ;    %  Image plot, Showing Cluster Results
                             %        Sorted Start Number vs. Case Number
                             %        Highlight those similar to most common 

        nfigopen = nfigopen + 1 ;
        figure(nfigopen) ;
        clf ;

        midxsc = midxs ;
        midxsc(visimilar,:) = midxsc(visimilar,:) + 2 * ones(length(visimilar),n) ;
             %  upshift by 2 the members of the similar set

        colormap([ones(1,3); zeros(1,3); [1 0 0 ]; [0 0 1]]) ;
        image(midxsc) ;
          if isempty(titlefontsize) ;
            title([titlestr 'Similar Colored']) ;
          else ;
            title([titlestr 'Similar Colored'], ...
                  'FontSize',titlefontsize) ;
          end ;
          if isempty(labelfontsize) ;
            xlabel('Case Number') ;
            ylabel('Sorted (on Cluster Index) Start Number') ;
          else ;
            xlabel('Case Number','FontSize',labelfontsize) ;
            ylabel('Sorted (on Cluster Index) Start Number','FontSize',labelfontsize) ;
          end ;

        if ~isempty(savestr) ;
          orient landscape ;
          print('-dpsc2',[savestr 'ImageClustSimCol.ps']) ;
        end ;


      end ;



      if sum(viplot(6:end) > 0.5) ;    %  then need to flip restarts, to make then all similar

        midxsf = midxs ;
            %  flipped version of midx
        midxsf(visimilar,:) = 3 * ones(length(visimilar),n) - midxsf(visimilar,:);
            %  mapping that gives:   1 --> 2,   2 --> 1


        if viplot(6) == 1 ;    %  Image plot, Showing Cluster Results
                               %        Sorted Start Number vs. Case Number
                               %        Classes Flipped, to be similar to most common

          nfigopen = nfigopen + 1 ;
          figure(nfigopen) ;
          clf ;

          colormap([ones(1,3); zeros(1,3)]) ;
          image(midxsf) ;
            if isempty(titlefontsize) ;
              title([titlestr 'Flipped Similar Cases']) ;
            else ;
              title([titlestr 'Flipped Similar Cases'], ...
                    'FontSize',titlefontsize) ;
            end ;
            if isempty(labelfontsize) ;
              xlabel('Case Number') ;
              ylabel('Sorted (on Cluster Index) Start Number') ;
            else ;
              xlabel('Case Number','FontSize',labelfontsize) ;
              ylabel('Sorted (on Cluster Index) Start Number','FontSize',labelfontsize) ;
            end ;

          if ~isempty(savestr) ;
            orient landscape ;
            print('-dps2',[savestr 'ImageClustFlip.ps']) ;
          end ;


        end ;



        if viplot(7) == 1 ;    %  Image plot, Showing Cluster Results
                                   %        Classes with common cases in Class 1
                                   %        Sorted Start Number vs. Sorted Case Number

          mflag2 = (midxsf == 2) ;
          vflag2 = sum(mflag2,1) ;
              %  number of 2s in each column
          [temp,vind] = sort(vflag2) ;
          midxsfs = midxsf(:,vind) ;
              %  sort columns by number of 2s

          nfigopen = nfigopen + 1 ;
          figure(nfigopen) ;
          clf ;

          colormap([ones(1,3); zeros(1,3)]) ;
          image(midxsfs) ;
            if isempty(titlefontsize) ;
              title([titlestr 'Rows & Cols Sorted']) ;
            else ;
              title([titlestr 'Rows & Cols Sorted'], ...
                    'FontSize',titlefontsize) ;
            end ;
            if isempty(labelfontsize) ;
              xlabel('Sorted (on # of 2s) Case Number') ;
              ylabel('Sorted (on Cluster Index) Start Number') ;
            else ;
              xlabel('Sorted (on # of 2s) Case Number','FontSize',labelfontsize) ;
              ylabel('Sorted (on Cluster Index) Start Number','FontSize',labelfontsize) ;
            end ;

          if ~isempty(savestr) ;
            orient landscape ;
            print('-dps2',[savestr 'ImageClustSortRC.ps']) ;
          end ;


        end ;


      end ;


    end ;


  end ;


end ; 


if nargout > 1 ;    %  Then need to create output variable

  bestCI = min(vindex) ;

end ;
