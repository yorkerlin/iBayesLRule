% main function to perform Stochastic Gradient Variational Bayes with gamma
% variational posteriors.
% logL - log likelihood function, second output arg is the gradient
% D - dimension of parameter space
% settings - various settings
% myCallback - can be used for plotting etc.
function [a,b,ll, settings]=gammaSGVB_bbvi( logL, D, settings, myCallback )

%adam parameters
epsilon=1e-8; decay_factor_mean=0.9; decay_factor_var=0.999;

%rng(100);
eps=1e-10;

%rng('shuffle');
tic

a=settings.inita;
b=settings.initb;

offset = 1e-2;
r=[log(exp(a-offset)-1.); log(exp(b)-1.)];

if settings.useAdam
    g_mean_r = 0.0*r;
    g_var_r = 0.0*r;
end


llCounter=0; % count likelihood evaluateion
ll=[];
if nargin()<=3
    ll=1;
    llCounter=1;
end

% test gradient evaluation using finite differences? useful for debugging
% but slow for big models
if settings.testGrad
    fprintf(1,'Testing gradients\n');
    tests=gamrnd(a,1./b);
    [l0,ag]=logL(tests);
%     ndeps=1e-6;
    for d=1:D
       ndeps=min( tests(d)*1e-6, 1e-6);
       temp=tests;
       temp(d)=temp(d)+ndeps;
       nd= ( logL(temp) - l0 ) / ndeps;
%        fprintf(1,'%f=%f\n',nd,ag(d));
       assert( abs(nd - ag(d) )/max(abs(nd),.001) < .05 );
    end
    fprintf(1,'Test grad passed\n');
end

ms=zeros(D*2,1) * (1-settings.g2blend);
if settings.useAdadelta
    msG=zeros(D*2,1);
    msX=zeros(D*2,1);
end
samplesUsed=zeros(settings.samples+1,1);
settings.it=zeros(settings.samples+1,1);
settings.train=zeros(settings.samples+1,1);
settings.test=zeros(settings.samples+1,1);
elboHistory=[];
settings
S=settings.nSamples;

global time_cache;
time_cache=zeros(settings.samples+1,1);
global is_cache;

tic;
for sampleIndex=1:(settings.samples+1)
    %samplesUsed(sampleIndex)=samplesUsed(sampleIndex)+1;

    logls=zeros(S,1);
    gs=zeros(2*D,S);

    for s=1:S
    [logl ghat]=evaluate_fun(settings, logL, eps, a, b, D, s*sampleIndex);
    %[logl ghat]=evaluate_fun2(settings, logL, eps, a, b, D, s*sampleIndex);
    logls(s)=logl;
    gs(:,s)=ghat;
    end

    logl=mean(logls);
    ghat=mean(gs,2);

    if is_cache==1
    elboHistory(end+1) = logl;

    rng(12467);
    ecov=zeros(settings.D,settings.D);
    nsamples=2000;
    for i=1:nsamples
	W= gamrnd(a , 1./b);
	EW=reshape(W ,[settings.D settings.K]);
	ecov=ecov+ EW * EW' + eye(settings.D) * settings.sigma2;
    end
    sgvbCov=ecov/nsamples;
    test=mean(mvnpdfl(settings.ytest', zeros(1,settings.D), sgvbCov));


    EW=reshape(a ./ b ,[settings.D settings.K]);
    sgvbCov=EW * EW' + eye(settings.D) * settings.sigma2;
    %train=mean(mvnpdfl(settings.ytrain', zeros(1,settings.D), sgvbCov));
    test2=mean(mvnpdfl(settings.ytest', zeros(1,settings.D), sgvbCov));
    %fprintf('%d) nlz=%.4f train=%.4f test=%.4f\n',sampleIndex,logl,train,test2);

    fprintf('%d) nlz=%.4f post-test=%.4f e-test=%.4f adam\n',sampleIndex,logl,test,test2);

    settings.it(sampleIndex)=sampleIndex;
    %settings.train(sampleIndex)=train;
    settings.test(sampleIndex)=test;
    end

    if sampleIndex == settings.samples+1
	break
    end

    if is_cache ==1
    if length(elboHistory)>10
       elboHistory(1)=[];
    end
    end

   ghat=ghat ./(1+exp(-r));
   assert( ~any(isnan(ghat)) )
   assert( ~any(isinf(ghat)) )


   if settings.useAdaGrad
       ms=settings.msBlend*ms+settings.g2blend*ghat.^2;
       ghat=ghat ./ (1e-6 + sqrt(ms) );
   end

   if settings.useAdadelta
       msG=settings.rho*ghat.^2+(1-settings.rho)*msG;
       ghat = ghat .* sqrt( msX + settings.eps ) ./  sqrt( msG + settings.eps );
   end

   if settings.useAdadelta
       msX=settings.rho*ghat.^2+(1-settings.rho)*msX;
   end

    if settings.useAdam
        [ghat,g_mean_r,g_var_r] = adam(ghat,g_mean_r,g_var_r,settings.stepSize,sampleIndex,decay_factor_mean,decay_factor_var,epsilon);
    else
       ghat =settings.stepSize*ghat;
    end

   r=r+ghat;

   a=log1pe(r(1:D))+offset;
   b=log1pe(r((D+1):end));

   if is_cache==1
   if mod(sampleIndex,20)==1
    llCounter=llCounter+1;
        ll(llCounter)=mean(elboHistory);
        if nargin()>3
            myCallback(a,b);
        end
       if settings.plot
           subplot(2,2,4);
           hold off;
           plot(ll);  drawnow();
       end
   end
   end

   assert(~any(isinf(a)));
   assert(~any(isinf(b)));
   assert(~any(isnan(a)));
   assert(~any(isnan(b)));

   time_cache(sampleIndex+1) = toc;
end


end

function [ll ghat]=evaluate_fun(settings, logL, eps, a, b, D, idx)
    [x, dfda, dfdb] = gammarnd_new(a,b,D,idx);
    [logl,glp]=logL(x); %without prior
    assert( settings.usePrior==0 );

	%fprintf('exact entropy and prior\n')
    a0=settings.inita;
    b0=settings.initb;
    %prior: E[ log p(x) ] = sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b) );
    ll_no_posterior_entropy=logl + sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b) );

    %ghat= [ dfda .* glp + ( 1.0 + (1-a) .* psi(1,a) ) + (a0-1).*psi(1,a) - b0./b ;
    %        dfdb .* glp - 1.0./b - (a0-1).*(1./b) + a.*(b0./(b.^2)) ] ;
    ghat= [ dfda .* glp + (a0-a) .* psi(1,a)  + (b-b0)./b ;
            dfdb .* glp - a0./b + (a.*b0)./(b.^2) ] ;

    ll = ll_no_posterior_entropy +sum( a - log(b) + gammaln(a) + (1-a).*psi(a) );

    assert( ~any(isnan(ghat)) )
    assert( ~any(isinf(ghat)) )
end


function [ll ghat]=evaluate_fun2(settings, logL, eps, a, b, D, idx)
rng(idx);
    x = gamrnd(a, 1./b);
    [logl]=logL(x);

    if settings.usePrior
      ghat= [ logl.*(log(b) - psi(a) + log(x)) + ( 1.0 + (1-a) .* psi(1,a) );
	      logl.*(a./b - x) - 1.0./b ];
      %ll=logl +sum( a - log(b) + gammaln(a) + (1-a).*psi(a) );
      ll_no_posterior_entropy = logl;
    else

      a0=settings.inita;
      b0=settings.initb;
      ghat = [logl.*(log(b) - psi(a) + log(x)) + ( 1.0 + (1-a) .* psi(1,a) ) + (a0-1).*psi(1,a) - b0./b ;
	      logl.*(a./b - x) - 1.0./b - (a0-1).*(1./b) + a.*(b0./(b.^2)) ];

      %ll=logl +sum( a - log(b) + gammaln(a) + (1-a).*psi(a) ) ...
      %+ sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b) );

      ll_no_posterior_entropy=logl + sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b) );
    end

    ll=ll_no_posterior_entropy +sum(a - log(b) + gammaln(a) + (1-a).*psi(a) );
end

