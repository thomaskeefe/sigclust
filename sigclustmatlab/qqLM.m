function [qqout,paramout] = qqLM(data,paramstruct) 
% QQ, Quantile - Quantile plot for testing distributional form
%      against standard distributions, currently including
%      Uniform(0,1), Gaussian (normal), log normal, Pareto, 
%      Pareto-log (still Pareto, but viewed on scale of log data),
%      Weibull, Weibull-log
%   Function Written by Lee, Mihee and Steve Marron
%   Replaces Marron's matlab function qqSM, which it
%   improves by including the Gamma and Beta distributions
% Inputs:
%   data        - n x 1 column vector of data
%   paramstruct - a Matlab structure of input parameters
%                    Use: "help struct" and "help datatypes" to
%                         learn about these.
%                    Create one, using commands of the form:
%
%       paramstruct = struct('field1',values1,...
%                            'field2',values2,...
%                                             ) ;
%
%                          where any of the following can be used,
%                          these are optional, misspecified values
%                          revert to defaults
%
%    fields            values
%
%    idist            0  Uniform  (allows "anything" by first applying
%                                      Probability Integral Transform)
%                     1  (default)  Gaussian
%                     2  Pareto
%                     3  Weibull
%                     4  Gumbel ("Double Exponential")
%                     5  Absolute Gaussian Power ("Max of Absolute Gaussians")
%                     6  Gamma  
%                     7  Beta  
%                     11 Lognormal (Classical distn: exp(N(mu,sig2)),
%                                      view on log-log scale)
%                     12 Logpareto (really Pareto, but log-log view)
%                     13 Logweibull (really Weibull, but log-log view)
%                     14 Loggumbel (really Gumbel, but log-log view) 
%                     15 Log Abs Gauss Pow (log-log view)
%                     16 Loggamma (really Gamma, but log-log view)
%
%            Associated Parameters (will estimate when not specified)
%
%            0.  Uniform
%                     No parameters allowed, only Unif(0,) is treated
%
%            1.  Gaussian:
%    mu               Mean 
%    sigma            Standard Deviation
%
%            2.  Pareto (moved to left, "to start at 0")
%                       (mean is (alpha * sigma) / (alpha - 1)  -  1):
%    alpha            Classical Tail Index
%    sigma            Scale Parameter
%
%            3.  Weibull (alpha is power of exponent)
%                        (mean is sigma * gamma(1/alpha + 1)):
%    alpha            Shape Parameter
%    sigma            Scale Parameter
%
%            4.  Gumbel:
%    mu               Location Parameter
%    sigma            Scale Parameter
%
%            5.  Power of Absolute Gaussian:
%    alpha            Shape Parameter (power, i.e. number of indep. Gaussians)
%    sigma            Scale Parameter
%
%            6. Gamma (mean is alpha*beta):
%    alpha            shape parameter 
%    beta             scale parameter
%          
%            7. Beta: 
%    alpha            Degree of polynomial at 0
%    beta             Degree of polynomial at 1
%
%
%    vqalign          vector of percentages for 2 quantiles to align
%                     (default:  [.5; .75])
%                         Won't work for Gamma and Beta, where
%                             Matlab functions gamfit and betafit are used
%                             to compute maximum likelihood fit
%
%    nsim             number of psuedo data sets to simulate,
%                              to display random variability
%                     (default = 100)
%                     0  no simulation, only straight QQ plot
%                         Note:  this creates an n x nsim matrix.
%                                When this is too big for memory, 
%                                then use a negative value (e.g. -100),
%                                for a slow, one at a time computation
%
%    simseed          seed for simulated overlay (to display randomness)
%                         default is [] (for using current Matlab seed) 
%                                (should be an integer with <= 8 digits)
%
%    nsimplotval      number of values to use in display of
%                     simulated versions to assess variability
%                     (useful to speed graphics with large
%                      sample sizes, uses nsimplotval/3 values
%                      at each end, and the rest equally spaced
%                      in the middle, ignored when ndata is
%                      smaller or when nsim = 0)
%                     0  use default nsimplotval = 900
%                         (tests suggest:
%                              3000  "no visual difference"
%                              900   "small visual difference"
%                              300   "larger visual difference"
%
%    icolor           1  (default)  full color version (for talks)
%                     0  fully black and white version (for papers)
%                            Note:  for icolor = 0, on many output
%                                   devices, nsim = 100 may overplot,
%                                   looks much nicer with nsim = 40
%
%    savestr          string controlling saving of output,
%                         either a full path, or a file prefix to
%                         save in matlab's current directory,
%                         will add .ps, and save as either
%                              color postscript (icolor = 1)
%                         or
%                              black&white postscript (when icolor = 0)
%                     unspecified:  results only appear on screen
%
%    titlestr         Title (default is 'Q-Q Plot')
%
%    titlefontsize    Font Size for title (uses Matlab default)
%                                   (18 is "fairly large")
%
%    xlabelstr        String for labeling x axes 
%                                  (default is Dist. Name + Q')
%
%    ylabelstr        String for labeling y axes 
%                                  (default is 'Data Q')
%
%    labelfontsize    Font Size for axis labels (uses Matlab default)
%                                   (18 is "fairly large")
%
%    ishowpar         0  (default) don't show parameters
%                     1  show parameters
%
%    parfontsize      Font Size for parameter text (uses Matlab default)
%                                   (18 is "fairly large", consider 15)
%
%    left             left end of plot range (default is min of data)
%
%    right            right end of plot range (default is max of data)
% 
%    bottom           bottom of plot range (default is min of data)
%
%    top              top of plot range (default is max of data)
% 
%    ioverlay         0  no overlay line
%                     1  (default) overlay 45 degree line
%                     2  overlay least squares fit line
%
%    ishowcross       0  (default)  don't show fit quantiles
%                     1  show quantiles where fit occurs
%
%    vshowq           []  (default)  don't show additional quantiles
%                     vector of probabilities:  show and label the
%                                  quantiles for these probabilities
%
%    iscreenwrite     0  (default)  don't write progress to screen
%                     1  write messages to screen to show progress
%
%    maxstep          maximum number of steps for Pareto Estimation
%                     (default = 1000) 
%
%    relaccthreshold  Relative Accuracy Threshold for
%                     Pareto Estimation iterations
%                     (default = 10^(-6))
%
%
% Outputs:
%     For nargout = 2:  graphics in current axes,
%                     & 2 column matrix, 
%                            theoretical quantiles in 1st column
%                            empirical quantiles in 2nd column
%                     & 2x1 vector with estimated parameters
%                            as either [mu; sigma] or [alpha; sigma]
%     For nargout = 1:  graphics in current axes,
%                     & 2 column matrix, 
%                            theoretical quantiles in 1st column
%                            empirical quantiles in 2nd column
%     For nargout = 0:  graphics in current axes
%     When savestr exists,   Postscript file saved in 'savestr'.ps
%                        (color postscript for icolor = 1)
%                        (B & W postscript for icolor = 0)
%
% Assumes path can find personal functions:
%     cquantSM.m



%    Copyright (c) Lee, Mihee,  J. S. Marron 2000-2008



%  First set parameters to defaults
%
idist = 1 ;
vqalign =[.5; .75] ;
nsim = 100 ;
simseed = [] ;
nsimplotval = 900 ;
icolor = 1 ;
savestr = [] ;
titlestr = 'Q-Q Plot' ;
titlefontsize = [] ;
xlabelstr = ', Q' ;
labelfontsize = [] ;
ylabelstr = 'Data Q' ;
ishowpar = 0 ;
parfontsize = [] ;
left = min(data) ;
right = max(data) ;
bottom = min(data) ;
top = max(data) ;
ioverlay = 1 ;
ishowcross = 0 ;
vshowq = [] ;
iscreenwrite = 0 ;
maxstep = 1000 ;
relaccthreshold = 10^(-6) ;




%  Now update parameters as specified,
%  by parameter structure (if it is used)
%
if nargin > 1 ;   %  then paramstruct has been added

  if isfield(paramstruct,'idist') ;    %  then change to input value
    idist = getfield(paramstruct,'idist') ; 
  end ;

  if isfield(paramstruct,'vqalign') ;    %  then change to input value
    vqalign = getfield(paramstruct,'vqalign') ; 
  end ;

  if isfield(paramstruct,'nsim') ;    %  then change to input value
    nsim = getfield(paramstruct,'nsim') ; 
  end ;

  if isfield(paramstruct,'simseed') ;    %  then change to input value
    simseed = getfield(paramstruct,'simseed') ; 
  end ;

  if isfield(paramstruct,'nsimplotval') ;    %  then change to input value
    nsimplotval = getfield(paramstruct,'nsimplotval') ; 
  end ;

  if isfield(paramstruct,'icolor') ;    %  then change to input value
    icolor = getfield(paramstruct,'icolor') ; 
  end ;

  if isfield(paramstruct,'savestr') ;    %  then change to input value
    savestr = getfield(paramstruct,'savestr') ; 
    if ~ischar(savestr) ;    %  then invalid input, so give warning
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from qqLM.m:       !!!') ;
      disp('!!!   Invalid savestr,           !!!') ;
      disp('!!!   using default of no save   !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      savestr = [] ;
    end ;
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

  if isfield(paramstruct,'ishowpar') ;    %  then change to input value
    ishowpar = getfield(paramstruct,'ishowpar') ; 
  end ;

  if isfield(paramstruct,'parfontsize') ;    %  then change to input value
    parfontsize = getfield(paramstruct,'parfontsize') ; 
  end ;

  if isfield(paramstruct,'left') ;    %  then change to input value
    left = getfield(paramstruct,'left') ; 
  end ;

  if isfield(paramstruct,'right') ;    %  then change to input value
    right = getfield(paramstruct,'right') ; 
  end ;

  if isfield(paramstruct,'bottom') ;    %  then change to input value
    bottom = getfield(paramstruct,'bottom') ; 
  end ;

  if isfield(paramstruct,'top') ;    %  then change to input value
    top = getfield(paramstruct,'top') ; 
  end ;

  if isfield(paramstruct,'ioverlay') ;    %  then change to input value
    ioverlay = getfield(paramstruct,'ioverlay') ; 
  end ;

  if isfield(paramstruct,'ishowcross') ;    %  then change to input value
    ishowcross = getfield(paramstruct,'ishowcross') ; 
  end ;

  if isfield(paramstruct,'vshowq') ;    %  then change to input value
    vshowq = getfield(paramstruct,'vshowq') ; 
  end ;

  if isfield(paramstruct,'iscreenwrite') ;    %  then change to input value
    iscreenwrite = getfield(paramstruct,'iscreenwrite') ; 
  end ;

  if isfield(paramstruct,'maxstep') ;    %  then change to input value
    maxstep = getfield(paramstruct,'maxstep') ; 
  end ;

  if isfield(paramstruct,'relaccthreshold') ;    %  then change to input value
    relaccthreshold = getfield(paramstruct,'relaccthreshold') ; 
  end ;


else ;   %  create a dummy structure
  paramstruct = struct('nothing',[]) ;


end ;  %  of resetting of input parameters





if  size(data,2) > 1  |  size(data,1) < 2 ;

  disp(' ') ;
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp('!!!   Error from qqLM.m:                  !!!') ;  
  disp('!!!   Input data must be a column vector  !!!') ;
  disp('!!!   Terminating Execution               !!!') ;  
  disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  disp(' ') ;

  return ;

end ;




%  Set internal parameters
%
if icolor == 1 ;
  colorcell = {'r' 'g' 'b'} ;
          %  1st - data
          %  2nd - line
          %  3rd - simulated versions
  ltypestr = '-' ;
  simltypestr = '-' ;
else ;
  colorcell = {'k' 'k' [0.2 0.2 0.2]} ;
  ltypestr = '--' ;
  simltypestr = ':' ;
end ;
paramout = [] ;



%  Get data quantiles
n = length(data) ;
qdata = sort(data) ;
          %  put in increasing order




%  Get theoretical quantiles
%
pgrid = (1:n)' / (n + 1) ;
ididqalign = 0 ;

if idist == 0 ;
  diststr = 'Uniform(0,1)' ;

  qtheory = pgrid ;

  paramout = [] ;
  par1str = '' ;
  par1val = [] ;
  par2str = '' ;
  par2val = [] ;

  if ishowpar == 1 ;
    disp('!!!   Warning from qqLM.m:  have reset ishowpar to 0') ;
    disp('!!!       since parameters not allowed for Unif(0,1)') ;
    ishowpar = 0 ;
  end ;

elseif  idist == 1  |  idist == 11  ;  
  if idist == 1 ;
    diststr = 'Gaussian' ;
  elseif idist == 11 ;
    diststr = 'LogNormal' ;

    if qdata(1) <= 0 ;
      disp(' ') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Error from qqLM.m:           !!!') ;  
      disp('!!!   Input data must be positive  !!!') ;
      disp('!!!   Terminating Execution        !!!') ;  
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp(' ') ;

      return ;
    end ;

    qdata = log(qdata) ;
  end ;

  if isfield(paramstruct, 'mu') &  isfield(paramstruct, 'sigma');
    if isfield(paramstruct, 'vqalign');
        disp(' ');
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
        disp('!!!   Warning from qqLM.m:                  !!!');
        disp('!!!   Parameters mu and sigma were input,   !!!');
        disp('!!!   so vqalign will be ignored            !!!');
        disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
        disp(' ');
    end;

    mu = getfield(paramstruct,'mu');
    sigma = getfield(paramstruct,'sigma');
  else;
    if isfield(paramstruct, 'vqalign');
      p1 = vqalign(1);
      p2 = vqalign(2);
      vq = cquantSM(qdata, vqalign,0);
           % 0 to indicate presorted data
      q1hat = vq(1);
      q2hat = vq(2);

      sigma = (q1hat - q2hat)/(norminv(p1) - norminv(p2));
      mu = q1hat - sigma * norminv(p1);
      ididqalign = 1;
    else;
      mu = mean(qdata);
      sigma = std(qdata);
    end;
  end;


  qtheory = mu + sigma * norminv(pgrid) ;


  paramout = [mu; sigma] ;

  par1str = '\mu' ;
  par1val = mu ;
  par2str = '\sigma' ;
  par2val = sigma ;


elseif  idist == 2  |  idist == 12  ;  
  if idist == 2 ;
    diststr = 'Pareto' ;
  elseif idist == 12 ;
    diststr = 'Pareto-log' ;
  end ;

  if qdata(1) <= 0 ;
    disp(' ') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from qqLM.m:           !!!') ;  
    disp('!!!   Input data must be positive  !!!') ;
    disp('!!!   Terminating Execution        !!!') ;  
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp(' ') ;

    return ;
  end ;

  if  isfield(paramstruct,'alpha')  & ...
          isfield(paramstruct,'sigma')  ;    %  then change to input value
    alpha = getfield(paramstruct,'alpha') ; 
    sigma = getfield(paramstruct,'sigma') ; 
  else ;    %  estimate from data

    p1 = vqalign(1) ;
    p2 = vqalign(2) ;
    vq = cquantSM(qdata,vqalign,0) ;
         %  0 to indicate presorted data
    q1hat = vq(1) ;
    q2hat = vq(2) ;

    %  Initiate iterative solution for "solving equations":
    %      p1 = F(q1hat)
    %      p2 = F(q2hat)
    numera = log(1 - p1) - log(1 - p2);
    oldsigma = 0 ;
          %  common and starting values

    for istep = 1:maxstep ;    %  loop through sigma values

      q1z = q1hat + oldsigma ;
      q2z = q2hat + oldsigma ;

      alpha = numera / (log(q2z) - log(q1z)) ;
      sigma = q1z * (1 - p1)^(1/alpha) ;

      relacc = abs(sigma - oldsigma) / sigma ;

      if  iscreenwrite == 1  &  floor(istep/50) == istep/50 ;
        disp(['     For iteration step ' num2str(istep) ...
                    ', relacc = ' num2str(relacc) ...
                    ', alpha = ' num2str(alpha) ...
                    ', sigma = ' num2str(sigma)]) ;
       end;

      
      if relacc < relaccthreshold ;    %  then have converged, so quit
        errflag = 0 ;
        if  iscreenwrite == 1  ;
          disp(['     For iteration step ' num2str(istep) ...
                      ', relacc = ' num2str(relacc) ...
                      ', alpha = ' num2str(alpha) ...
                      ', sigma = ' num2str(sigma)]) ;
        end;
        break ;
      else ;    %  not close enough yet, continue
        errflag = 1 ;
        oldsigma = sigma ;
      end ;

    end ;    %  of istep loop through iterated alpha values

    if errflag ~= 0 ;
      disp('!!!   Warning from qqLM.m:  Pareto fit may be unstable   !!!') ;
    end ;

    ididqalign = 1 ;
  end ;


  qtheory = ((1 - pgrid).^(-1/alpha) - 1) * sigma ;



  if idist == 12 ;    %  then need to fix scales
    qdata = log(qdata) ;
    qtheory = log(qtheory) ;
    if ididqalign == 1 ;    %  then have done quantile matching
      q1hat = log(q1hat) ;
      q2hat = log(q2hat) ;
    end ;
  end ;


  paramout = [alpha; sigma] ;

  par1str = '\alpha' ;
  par1val = alpha ;
  par2str = '\sigma' ;
  par2val = sigma ;


elseif  idist == 3  |  idist == 13  ;
  if idist == 3 ;
    diststr = 'Weibull' ;
  elseif idist == 13 ;
    diststr = 'Weibull-log' ;
  end ;

  if qdata(1) <= 0 ;
    disp(' ') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from qqLM.m:           !!!') ;  
    disp('!!!   Input data must be positive  !!!') ;
    disp('!!!   Terminating Execution        !!!') ;  
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp(' ') ;

    return ;
  end ;

  if  isfield(paramstruct,'alpha')  & ...
          isfield(paramstruct,'sigma')  ;    %  then change to input value
    alpha = getfield(paramstruct,'alpha') ; 
    sigma = getfield(paramstruct,'sigma') ; 
  else ;    %  estimate from data
    p1 = vqalign(1) ;
    p2 = vqalign(2) ;
    vq = cquantSM(qdata,vqalign,0) ;
         %  0 to indicate presorted data
    q1hat = vq(1) ;
    q2hat = vq(2) ;

    alpha = (log(-log(1 - p1)) - log(-log(1 - p2))) / ...
                      (log(q1hat) - log(q2hat)) ;
    sigma = exp(log(q1hat) - log(-log(1 - p1)) / alpha) ;
        %  These come from "solving equations":
        %      p1 = F(q1hat)
        %      p2 = F(q2hat)

    ididqalign = 1 ;
  end ;


  qtheory = sigma * (-log(1 - pgrid)).^(1/alpha) ;



  if idist == 13 ;    %  then need to fix scales
    qdata = log(qdata) ;
    qtheory = log(qtheory) ;
    if ididqalign == 1 ;    %  then have done quantile matching
      q1hat = log(q1hat) ;
      q2hat = log(q2hat) ;
    end ;
  end ;


  paramout = [alpha; sigma] ;

  par1str = '\alpha' ;
  par1val = alpha ;
  par2str = '\sigma' ;
  par2val = sigma ;

  
elseif  idist == 4  |  idist == 14  ;
  if idist == 4 ;
    diststr = 'Gumbel' ;
  elseif idist == 14 ;
    diststr = 'Gumbel-log' ;
  end ;

  if  isfield(paramstruct,'mu')  & ...
          isfield(paramstruct,'sigma')  ;    %  then change to input value
    mu = getfield(paramstruct,'mu') ; 
    sigma = getfield(paramstruct,'sigma') ; 
  else ;    %  estimate from data

    p1 = vqalign(1) ;
    p2 = vqalign(2) ;
    vq = cquantSM(qdata,vqalign,0) ;
         %  0 to indicate presorted data
    q1hat = vq(1) ;
    q2hat = vq(2) ;

    mu = (log(-log(p1)) * q2hat - log(-log(p2)) * q1hat) / ...
                   (log(-log(p1)) - log(-log(p2))) ;
    sigma = (q2hat - q1hat) / ...
                   (log(-log(p1)) - log(-log(p2))) ;
        %  These come from "solving equations":
        %      p1 = F(q1hat)
        %      p2 = F(q2hat)

    ididqalign = 1 ;
  end ;


  qtheory = -sigma * log(-log(pgrid)) + mu ;


  if idist == 14 ;    %  then need to fix scales
    qdata = log(qdata) ;
    qtheory = log(qtheory) ;
    if ididqalign == 1 ;    %  then have done quantile matching
      q1hat = log(q1hat) ;
      q2hat = log(q2hat) ;
    end ;
  end ;


  paramout = [mu; sigma] ;

  par1str = '\mu' ;
  par1val = mu ;
  par2str = '\sigma' ;
  par2val = sigma ;


elseif  idist == 5  |  idist == 15  ;  
  if idist == 5 ;
    diststr = 'AbsGauPow' ;
  elseif idist == 15 ;
    diststr = 'AbGaPo-log' ;
  end ;

  if qdata(1) <= 0 ;
    disp(' ') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from qqLM.m:           !!!') ;  
    disp('!!!   Input data must be positive  !!!') ;
    disp('!!!   Terminating Execution        !!!') ;  
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp(' ') ;

    return ;
  end ;

  if  isfield(paramstruct,'alpha')  & ...
          isfield(paramstruct,'sigma')  ;    %  then change to input value
    alpha = getfield(paramstruct,'alpha') ; 
    sigma = getfield(paramstruct,'sigma') ; 
  else ;    %  estimate from data
    p1 = vqalign(1) ;
    p2 = vqalign(2) ;

    if p2 <= p1 ;
      disp(' ') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Error from qqLM.m:                 !!!') ;  
      disp('!!!   Requires vqalign(1) < vqalign(2)   !!!') ;
      disp('!!!   Terminating Execution              !!!') ;  
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp(' ') ;

      return ;
    end ;

    vq = cquantSM(qdata,vqalign,0) ;
         %  0 to indicate presorted data
    q1hat = vq(1) ;
    q2hat = vq(2) ;

    %  Initiate iterative solution for finding sigma
    %      (can get alpha, by solving)
    sigmaold = median(qdata) ;
    maxstepsize = sigmaold ;
    loval = 0 ;
    hival = +Inf ;
    flag = logical(0) ;

    for istep = 1:maxstep ;    %  loop through sigma values

      sq1 = q1hat / (sigmaold * sqrt(2)) ;
      sq2 = q2hat / (sigmaold * sqrt(2)) ;
      if erf(sq1) < 1 - 10^(-12) ;
        f = (log(erf(sq2)) / log(erf(sq1))) - ...
                            (log(p2) / log(p1)) ;
        fprime = ((exp(-sq1^2) * sqrt(2/pi) * q1hat * log(erf(sq2))) / ...
                      (sigmaold^2 * erf(sq1) * log(erf(sq1))^2)) - ...
                      ((exp(-sq2^2) * sqrt(2/pi) * q2hat) / ...
                       (sigmaold^2 * erf(sq2) * log(erf(sq1)))) ;
        sigmanewton = sigmaold - f / fprime ;
            %  Compute Newton step

        if f > 0 ;
          hival = sigmaold ;
          flag = logical(1) ;
        else ; 
          loval = sigmaold ;
        end ;

        if flag ;    % then are above 0 with f
 
          if  sigmanewton < loval  |  sigmanewton > hival ;
                    %  then are outside range so take bisection step
            sigma = (hival + loval) / 2 ;
          else ;    %  then have good Newton step, so keep
            sigma = sigmanewton ;
          end ;
 
        else ;    %  then are below 0 in f value

          if sigmanewton - sigmaold < maxstepsize ;
            sigma = sigmanewton ;
          else ;
            sigma = sigmaold + maxstepsize ;
            maxstepsize = maxstepsize * 2 ;
          end ;
        end ;

      else ;
        sigma = 2 * sigmaold ;
        maxstepsize = maxstepsize * 2 ;
      end;
      


      relacc = abs(sigma - sigmaold) / sigma ;

      if  iscreenwrite == 1  &  floor(istep/50) == istep/50 ;
        disp(['     For iteration step ' num2str(istep) ...
                    ', relacc = ' num2str(relacc) ...
                    ', sigma = ' num2str(sigma)]) ;
       end;

      
      if relacc < relaccthreshold ;    %  then have converged, so quit
        errflag = 0 ;
        if  iscreenwrite == 1  ;
          disp(['     For iteration step ' num2str(istep) ...
                      ', relacc = ' num2str(relacc) ...
                      ', sigma = ' num2str(sigma)]) ;
        end;
        break ;
      else ;    %  not close enough yet, continue
        errflag = 1 ;
        sigmaold = sigma ;
      end ;

    end ;    %  of istep loop through iterated alpha values

    if errflag ~= 0 ;
      disp('!!!   Warning from qqLM.m:  Mas Abs Gaussian fit may be unstable   !!!') ;
    end ;

    ididqalign = 1 ;

    alpha = (( log(p1) / log(erf(q1hat / (sqrt(2) * sigma))) ) + ...
            ( log(p2) / log(erf(q2hat / (sqrt(2) * sigma))) )) / 2 ;
  end;

  qtheory = norminv((1 + pgrid.^(1/alpha)) / 2) * sigma ;


  if idist == 15 ;    %  then need to fix scales
    qdata = log(qdata) ;
    qtheory = log(qtheory) ;
    if ididqalign == 1 ;    %  then have done quantile matching
      q1hat = log(q1hat) ;
      q2hat = log(q2hat) ;
    end ;
  end ;


  paramout = [alpha; sigma] ;

  par1str = '\alpha' ;
  par1val = alpha ;
  par2str = '\sigma' ;
  par2val = sigma ;

  
elseif  idist == 6  |  idist == 16  ;
  if idist == 6 ;
    diststr = 'Gamma' ;
  elseif idist == 16 ;
    diststr = 'Gamma-log' ;
  end ;

  if qdata(1) <= 0 ;
    disp(' ') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from qqLM.m:           !!!') ;  
    disp('!!!   Input data must be positive  !!!') ;
    disp('!!!   Terminating Execution        !!!') ;  
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp(' ') ;

    return ;
  end ;

  if isfield(paramstruct, 'vqalign');
      disp(' ');
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
      disp('!!!   Warning from qqLM.m:                         !!!');
      disp('!!!   vqalign not supported for Gamma (-log) fit   !!!');
      disp('!!!   will use MLE fit instead                     !!!');
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
      disp(' ');
  end;

  if  isfield(paramstruct,'alpha')  &  isfield(paramstruct,'beta')  ;
                   %  then change to input value
    alpha = getfield(paramstruct,'alpha') ; 
    beta = getfield(paramstruct,'beta') ; 

  else; %  estimate from data
%         alpha = (mean(data))^2/var(data);
%         beta = var(data)/mean(data);     
    parahat = gamfit(data);
    alpha = parahat(1);
    beta = parahat(2);
  end ;

  qtheory = gaminv(pgrid, alpha, beta) ;
  paramout = [alpha; beta] ;

  if idist == 16 ;    %  then need to fix scales
    qdata = log(qdata) ;
    qtheory = log(qtheory) ;
    if ididqalign == 1 ;    %  then have done quantile matching
      q1hat = log(q1hat) ;
      q2hat = log(q2hat) ;
    end ;
  end ;

  par1str = '\alpha' ;
  par1val = alpha ;
  par2str = '\beta' ;
  par2val = beta ;  


elseif idist == 7;
  diststr = 'Beta';

  if qdata(1) <= 0 ;
    disp(' ') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from qqLM.m:           !!!') ;  
    disp('!!!   Input data must be positive  !!!') ;
    disp('!!!   Terminating Execution        !!!') ;  
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp(' ') ;
    return ;
  end ;

  if qdata(end) > 1 ;
    disp(' ') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Error from qqLM.m:         !!!') ;  
    disp('!!!   Input data must be <= 1    !!!') ;
    disp('!!!   Terminating Execution      !!!') ;  
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp(' ') ;
    return ;
  end ;

  if isfield(paramstruct, 'vqalign');
      disp(' ');
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
      disp('!!!   Warning from qqLM.m:                 !!!');
      disp('!!!   vqalign not supported for Beta fit   !!!');
      disp('!!!   will use MLE fit instead             !!!');
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
      disp(' ');
  end;

  if  isfield(paramstruct, 'alpha')  &  isfield(paramstruct,'beta')  ;
                   %  then change to input value
    alpha =  getfield(paramstruct, 'alpha');
    beta = getfield(paramstruct, 'beta');

  else; %  estimate from data
%         alpha = mean(data)*(mean(data)*(1-mean(data))-var(data))/var(data);
%         beta = (1-mean(data))*(mean(data)*(1-mean(data))-var(data))/var(data); 
    parahat = betafit(data);
    alpha = parahat(1);
    beta = parahat(2);
  end ;

  qtheory = betainv(pgrid, alpha, beta);
  paramout = [alpha; beta];

  par1str = '\alpha';
  par1val = alpha;
  par2str = '\beta';
  par2val = beta;
	
  
end ;    %  of idist if block




if iscreenwrite == 1 ;
  disp(['   Comparing with ' diststr ' distribution']) ;
end;


if strcmp(xlabelstr,', Q') ;    %  default value
  xlabelstr = [diststr, xlabelstr] ;
end ;


if strcmp(ylabelstr,'Data Q') ;    %  default value
  if idist > 10 ;    %  working on log scale
    ylabelstr = ['log ' ylabelstr] ;
  end ;
end ;


if idist > 10 ;

  if bottom == min(data) ;
    bottom = log(bottom) ;
  end ;

  if top == max(data) ;
    top = log(top) ;
  end ;

end ;


if ~isfield(paramstruct,'left') ;    %  then change to theoretical left
  left = min(qtheory) ; 
end ;

if ~isfield(paramstruct,'right') ;    %  then change to theoretical right
  right = max(qtheory) ; 
end ;






%  Make main graphic in current axes
%
plot(qtheory,qdata, ...
    [colorcell{1} '-'], ...
    'LineWidth',3) ;
  axis equal ;
  axis([left,right,bottom,top]) ;

  th = title(titlestr) ;
  if ~isempty(titlefontsize) ;
    set(th,'FontSize',titlefontsize) ;
  end ;

  xlh = xlabel(xlabelstr) ;    
  ylh = ylabel(ylabelstr) ;    
  if ~isempty(labelfontsize) ;
    set(xlh,'FontSize',labelfontsize) ;
    set(ylh,'FontSize',labelfontsize) ;
  end ;




%  Add lines (if needed)
%
if ioverlay == 1 ;    %  then overlay 45 degree line
  minmin = min(left,bottom) ;
  maxmax = max(right,top) ;
  hold on ;
    plot([minmin,maxmax],[minmin,maxmax], ...
         [colorcell{2} ltypestr], ...
         'LineWidth',2) ;
  hold off ;

elseif ioverlay == 2 ;    %  then overlay least squares fit line
  minmin = min(left,bottom) ;
  maxmax = max(right,top) ;
  lincoeffs = polyfit(qtheory,qdata,1) ;
          %  coefficients of simple least squares fit
  lineval = polyval(lincoeffs,[minmin;maxmax]) ;
  hold on ;
    plot([minmin;maxmax],lineval, ...
         [colorcell{2} ltypestr], ...
         'LineWidth',2) ;
  hold off ; 

end ;





%  Add simulated realizations (if needed)
%
if nsim > 0 ;

  if iscreenwrite == 1 ;
    disp('      generating simulated data') ;
  end ;


  if idist == 0 ;    %  Unif(0,1)

    if ~isempty(simseed) ;
      rand('seed',simseed) ;
    end ;
          %  set seed
    msimdat = rand(n,nsim) ;


  elseif  idist == 1  |  idist == 11  ;    %  Gaussian

    if ~isempty(simseed) ;
      randn('seed',simseed) ;
    end ;
          %  set seed
    msimdat = mu + sigma * randn(n,nsim) ;


  elseif  idist == 2  |  idist == 12  ;    %  Pareto

    if ~isempty(simseed) ;
      rand('seed',simseed) ;
    end ;
          %  set seed
    msimdat = sigma * ((1 - rand(n,nsim)).^(-1/alpha) - 1) ;

    if idist == 12 ;    %  then need to fix scales
      msimdat = log(msimdat) ;
    end ;


  elseif  idist == 3  |  idist == 13  ;    %  Weibull

    if ~isempty(simseed) ;
      rand('seed',simseed) ;
    end ;
          %  set seed
    msimdat = sigma * (-log(1 - rand(n,nsim))).^(1 / alpha) ;

    if idist == 13 ;    %  then need to fix scales
      msimdat = log(msimdat) ;
    end ;

  elseif  idist == 4  |  idist == 14  ;    %  Gumbel

    if ~isempty(simseed) ;
      rand('seed',simseed) ;
    end ;
          %  set seed
    msimdat = -sigma * log(-log(rand(n,nsim))) + mu ;

    if idist == 14 ;    %  then need to fix scales
      msimdat = log(msimdat) ;
    end ;


  elseif  idist == 5  |  idist == 15  ;    %  Maximun Abs Gaussian

    if ~isempty(simseed) ;
      rand('seed',simseed) ;
    end ;
          %  set seed

    msimdat = norminv((1 + rand(n,nsim).^(1/alpha)) / 2) * sigma ;

    if idist == 15 ;    %  then need to fix scales
      msimdat = log(msimdat) ;
    end ;

    
  elseif  idist == 6  | idist == 16  ;

    if ~isempty(simseed);
        rand('seed',simseed);
    end;
        %  set seed

    msimdat = gaminv(rand(n,nsim), alpha, beta);


    if idist == 16 ;    %  then need to fix scales
      msimdat = log(msimdat) ;
    end ;

 
  elseif  idist == 7  ;

    if ~isempty(simseed);
        rand('seed',simseed);
    end;
        %  set seed

    msimdat = betainv(rand(n,nsim), alpha, beta);


  end ;    %  of idist block



  if iscreenwrite == 1 ;
    disp('      sorting simulated data') ;
  end ;
  msimdat = sort(msimdat) ;
          %  sort each column


  if nsimplotval < n ;    %  then get reduced version for 
                          %  efficient plotting

    nspvo3 = floor(nsimplotval / 3) ;
          %  one third of total, to put at each end
    vindleft = (1:nspvo3)' ;
          %  left end  (include all)
    vindright = ((n-nspvo3+1):n)' ;
          %  right end  (include all)

    nspvlo = nsimplotval - length(vindleft) ...
                         - length(vindright) ;
          %  number of grid points left over (for use in middle)
    vindmid = round(linspace(nspvo3+1,n-nspvo3,nspvlo)') ;

    vind = [vindleft; vindmid; vindright] ;
          %  vector of indices, full

    qtheoryp = qtheory(vind,:) ;
    msimdatp = msimdat(vind,:) ;

  else ;  

    qtheoryp = qtheory ;
    msimdatp = msimdat ;

  end ;


  hold on  ;
    plot(qtheoryp,msimdatp,simltypestr,'Color',colorcell{3}) ;

    %  replot important stuff
    %
    plot(qtheory,qdata, ...
         [colorcell{1} '-'], ...
         'LineWidth',3) ;
    if ioverlay == 1 ;    %  then overlay 45 degree line
      plot([minmin,maxmax],[minmin,maxmax], ...
           [colorcell{2} ltypestr], ...
           'LineWidth',2) ;
    elseif ioverlay == 2 ;    %  then overlay least squares fit line
      plot([minmin;maxmax],lineval, ...
           [colorcell{2} ltypestr], ...
           'LineWidth',2) ;
    end ;


  hold off ;


elseif nsim < 0 ;    %  then compute and plot these individually
                     %  to avoid memory problems


  if ~isempty(simseed) ;
    rand('seed',simseed) ;
    randn('seed',simseed) ;
  end ;
      %  set seed

  pnsim = abs(nsim) ;    %  positive version of number to simulate
  
  for isim = 1:pnsim ;    %  loop through simulation steps

    if iscreenwrite == 1 ;
      disp(['      working on simulated data set ' num2str(isim) ...
                                            ' of ' num2str(pnsim)]) ;
    end ;

    if  idist == 1  |  idist == 11  ;    %  Gaussian
      vsimdat = mu + sigma * randn(n,1) ;
    elseif  idist == 2  |  idist == 12  ;    %  Pareto
      vsimdat = sigma * ((1 - rand(n,1)).^(-1/alpha) - 1) ;
      if idist == 12 ;    %  then need to fix scales
        vsimdat = log(vsimdat) ;
      end ;
    elseif  idist == 3  |  idist == 13  ;    %  Weibull
      vsimdat = sigma * (-log(1 - rand(n,1))).^(1 / alpha) ;
      if idist == 13 ;    %  then need to fix scales
        vsimdat = log(vsimdat) ;
      end ;
    elseif  idist == 4  |  idist == 14  ;    %  Gumbel
      vsimdat = -sigma * log(-log(rand(n,1))) + mu ;
      if idist == 14 ;    %  then need to fix scales
        vsimdat = log(vsimdat) ;
      end ;
    elseif  idist == 5  |  idist == 15  ;    %  max power Gaussian
      vsimdat = norminv((1 + rand(n,1).^(1/alpha)) / 2) * sigma ;
      if idist == 15 ;    %  then need to fix scales
        vsimdat = log(vsimdat) ;
      end ;
    end ;    %  of idist block
    vsimdat = sort(vsimdat) ;
          %  sort this column


    if nsimplotval < n ;    %  then get reduced version for 
                            %  efficient plotting

      nspvo3 = floor(nsimplotval / 3) ;
            %  one third of total, to put at each end
      vindleft = (1:nspvo3)' ;
            %  left end  (include all)
      vindright = ((n-nspvo3+1):n)' ;
            %  right end  (include all)

      nspvlo = nsimplotval - length(vindleft) ...
                           - length(vindright) ;
            %  number of grid points left over (for use in middle)
      vindmid = round(linspace(nspvo3+1,n-nspvo3,nspvlo)') ;

      vind = [vindleft; vindmid; vindright] ;
            %  vector of indices, full

      qtheoryp = qtheory(vind,:) ;
      vsimdatp = vsimdat(vind,:) ;

    else ;  

      qtheoryp = qtheory ;
      vsimdatp = vsimdat ;

    end ;


    hold on  ;
      plot(qtheoryp,vsimdatp,simltypestr,'Color',colorcell{3}) ;
    hold off ;


  end ;    %  of loop through simulated data sets



  %  replot important stuff
  %
  hold on ;
    plot(qtheory,qdata, ...
         [colorcell{1} '-'], ...
         'LineWidth',3) ;
    if ioverlay == 1 ;    %  then overlay 45 degree line
      plot([minmin,maxmax],[minmin,maxmax], ...
           [colorcell{2} ltypestr], ...
           'LineWidth',2) ;
    elseif ioverlay == 2 ;    %  then overlay least squares fit line
      plot([minmin;maxmax],lineval, ...
           [colorcell{2} ltypestr], ...
           'LineWidth',2) ;
    end ;
  hold off ;



end ;    %  of simulated data if block




%  Add text for parameters (if needed)
%
if ishowpar == 1 ;
    tx = left + 0.6 * (right - left) ;
    ty = bottom + 0.25 * (top - bottom) ;
  th = text(tx,ty,[par1str ' = ' num2str(par1val)]) ;
  if ~isempty(parfontsize) ;
    set(th,'FontSize',parfontsize) ;
  end ;

    ty = bottom + 0.1 * (top - bottom) ;
  th = text(tx,ty,[par2str ' = ' num2str(par2val)]) ;
  if ~isempty(parfontsize) ;
    set(th,'FontSize',parfontsize) ;
  end ;

end ;





%  show fit quantiles (if needed)
%
if ishowcross == 1 ;

  if ididqalign == 1 ;    %  then show crossing quantiles
    hold on ;
      sch = plot([q1hat; q2hat], [q1hat; q2hat], 'ko') ;
    hold off ;
  else ;    %  then didn't actually do quantile fit, give warning
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
    disp('!!!   Warning from qqLM:                          !!!') ;
    disp('!!!   Didn''t compute parameters from quantiles,   !!!') ;
    disp('!!!   So won''t show crossing points               !!!') ;
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
  end ;

end ;    %  of ishowcross if-block





%  show additional quantiles (if needed)
%
if ~isempty(vshowq) ;

  for isq = 1:length(vshowq) ;
  
    sq = vshowq(isq) ;
    
    if  0 < sq   &  sq < 1  ;    %  then have valid probability

      [temp,qi] = min(abs((((1:n)' - 0.5) / n) - sq)) ;
          %  gets index of this quantile
          %  i.e. where prob is closest to sq prob

      qehat = qdata(qi) ;
      qthat = qtheory(qi) ;

      qhx = qthat + 0.02 * (right - left) ;

      hold on ;
        plot([qthat], [qehat], 'k+') ;
        th = text(qhx,qehat,[num2str(sq) ' quantile']) ;
        if ~isempty(parfontsize) ;
          set(th,'FontSize',parfontsize) ;
        end ;
      hold off ;

    else ;    %  then don't have valid probability, give error message

      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;
      disp('!!!   Warning from qqLM:                       !!!') ;
      disp('!!!   Invalid entry (not in (0,1)) in vshowq   !!!') ;
      disp('!!!   So won''t show this quantile              !!!') ;
      disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') ;

    end ;    %  of sq if-block  


  end ;    %    of isq loop through quantiles to show


end ;    %  of showq if-block





%  Save results (if needed)
%
if ~isempty(savestr) ;     %  then save results

  if iscreenwrite == 1 ;
    disp('    qqLM.m saving results') ;
  end ;



  orient landscape ;

  if icolor ~= 0 ;     %  then make color postscript
    print('-dpsc', [savestr '.ps']) ;
  else ;                %  then make black and white
    print('-dpsc', [savestr '.ps']) ;
  end ;



  if iscreenwrite == 1 ;
    disp('    qqLM.m finished save') ;
    disp('  ') ;
  end ;

end ;





%  create output (if needed)
%
if nargout > 0 ;

  qqout = [qtheory,qdata] ;

end ;


