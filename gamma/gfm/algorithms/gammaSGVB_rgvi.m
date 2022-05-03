% main function to perform Stochastic Gradient Variational Bayes with gamma
% variational posteriors.
% logL - log likelihood function, second output arg is the gradient
% D - dimension of parameter space
% settings - various settings
% myCallback - can be used for plotting etc.
function [a,b,ll,settings]=gammaSGVB_rgvi( logL, D, settings, myCallback )

eps=1e-10;

%rng('shuffle');
rng(100);

%prior
prior_a=settings.inita;
prior_b=settings.initb;

a=prior_a;
b=prior_b;

eta1 = a;
eta2 = b./a;

offset = 1e-2;

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

elboHistory=[];
settings
S=settings.nSamples;
samplesUsed=zeros(settings.samples+1,1);
settings.it=zeros(settings.samples+1,1);
settings.train=zeros(settings.samples+1,1);
settings.test=zeros(settings.samples+1,1);

global time_cache;
time_cache=zeros(settings.samples+1,1);
global is_cache;


tic
for sampleIndex=1:(settings.samples+1)
    samplesUsed(sampleIndex)=samplesUsed(sampleIndex)+1;

    logls=zeros(S,1);
    gs=zeros(2*D,S);

    for s=1:S
    [logl ghat]=evaluate_fun(settings, logL, eps, a, b, D, s*sampleIndex);
    %[logl ghat]=evaluate_fun2(settings, logL, eps, a, b, D, s*sampleIndex);
    logls(s)=logl;
    gs(:,s)=ghat;
    end

    %{
    logls2=zeros(S,1);
    gs2=zeros(2*D,S);
    parfor s=1:S
    [logl ghat]=evaluate_fun2(settings, logL, eps, a, b, D, s*sampleIndex);
    logls2(s)=logl;
    gs2(:,s)=ghat;
    end
    %}

    %{
    tt1=mean(logls); tt2=mean(logls2);
    fprintf('debug %.4f %.4f diff=%.4f\n', tt1, tt2, abs(tt1-tt2))
    %}

    logl=mean(logls);
    ghat=mean(gs,2);
    ga = ghat(1:D);
    gb = ghat((D+1):end);
    ghat(1:D) = ga + gb.*eta2;
    ghat((D+1):end) = gb.*eta1;

    assert( ~any(isnan(ghat)) )
    assert( ~any(isinf(ghat)) )


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
    %fprintf('%d) nlz=%.4f train=%.4f test=%.4f\n',sampleIndex,logl,train,test);

    fprintf('%d) nlz=%.4f post-test=%.4f e-test=%.4f rgvi-new\n',sampleIndex,logl,test,test2);
    settings.it(sampleIndex)=sampleIndex;
    %settings.train(sampleIndex)=train;
    settings.test(sampleIndex)=test;
    end

    if sampleIndex == settings.samples+1
	    break
    end

    if is_cache==1
    if length(elboHistory)>10
       elboHistory(1)=[];
    end
    assert(length(ghat) == 2*D);
    end

    d = zeros(2*D,1);
    for jj=1:D
        %I = [psi(1,eta1(jj))-1.0./eta1(jj), 0; 0, eta1(jj)/(eta2(jj).^2)];
        %tmp= I\[ghat(jj);ghat(jj+D)];
        %d(jj)=tmp(1); d(jj+D)=tmp(2);

        d(jj) = ghat(jj) / (psi(1,eta1(jj))-1.0./eta1(jj));
        d(jj+D) = ghat(jj+D) / ( eta1(jj)/(eta2(jj).^2) );
    end

    deta1=d(1:D); deta2=d(D+1:D+D);

    stepSize = settings.stepSize;

    %(e1**2*polygamma(2, e1) + 1)/(2*e1*(e1*polygamma(1, e1) - 1))
    %-1/e2

   stepSizesq2 =  stepSize*stepSize/2.;
   assert( all( ( (eta1.*psi(2, eta1)+1.0./eta1)./(2.0.*(eta1.*psi(1, eta1) -1.0)) ) <-1.0./eta1 ) )
   eta1 = eta1 + stepSize .* deta1 - stepSizesq2.*(deta1.*deta1).*((eta1.*psi(2, eta1)+1.0./eta1)./(2.0.*(eta1.*psi(1,eta1) -1.0))) ;
   eta2 = eta2 + stepSize .* deta2 - stepSizesq2.*(-deta2.*deta2./eta2) ;

   assert( all(eta1>0.) )
   assert( all(eta2>0.) )

   a = eta1;
   b = eta1.*eta2;

   assert( all(a>offset) )
   assert( all(b>0.) )

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

   assert( ~any(isinf(a)) & ~any(isinf(b)));
   assert( ~any(isnan(a)) & ~any(isnan(b) ));

   time_cache(sampleIndex+1) = toc;
end

end


function [ll ghat]=evaluate_fun(settings, logL, eps, a, b, D, idx)
    [x, dfda, dfdb] = gammarnd_new(a,b,D,idx);
    [logl,glp]=logL(x);% without prior

    a0=settings.inita;
    b0=settings.initb;

    %prior: sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b)
    %entropy: sum( a - log(b) + gammaln(a) + (1-a).*psi(a) )
    ll=logl +sum( a - log(b) + gammaln(a) + (1-a).*psi(a) ) ...
      + sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b) );

    %ghat= [ dfda .* glp ;
            %dfdb .* glp] ;%grad_likelihood

    ghat= [ dfda .* glp + (a0-a) .* psi(1,a)  + (b-b0)./b ;
            dfdb .* glp - a0./b + (a.*b0)./(b.^2) ] ; %grad_like_prior_entry

    assert( ~any(isnan(ghat)) )
    assert( ~any(isinf(ghat)) )
end


function [ll ghat]=evaluate_fun2(settings, logL, eps, a, b, D, idx)
rng(idx);
    x = gamrnd(a, 1./b);
    [logl]=logL(x);


      a0=settings.inita;
      b0=settings.initb;
      ghat = [logl.*(log(b) - psi(a) + log(x)) ;
              logl.*(a./b - x) ];

      ll=logl +sum( a - log(b) + gammaln(a) + (1-a).*psi(a) ) ...
      + sum ( (a0-1).*psi(a) - gammaln(a0) + log(b0) + (a0-1).*( log(b0)-log(b) ) - a.*(b0./b) );

end
