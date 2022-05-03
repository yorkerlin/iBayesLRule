% range of sample sizes
%ss=[10 20 30 40 50 100 200 500 1000 2000 5000 10000 ]; 
ss=[1000]; 

D=50;
K=10; 


rng(3345)
wtrue=(rand(D,K)<.2) .*  rand(D,K); %abs(randn(D,K)); gamrnd(1,1,D,K);

priora=1.0;
priorb=1.0;
%parfor rep=1:length(ss)
for rep=1:length(ss)

    N=ss(rep); 
    x=randn(K,N); 
    
    sigma2=.1; 
    y=wtrue * x + sqrt(sigma2) * randn(D,N); 

    % gammaSGVB GPFA ---------------
    %settings=gammaSGVBsettings(6); 


    beta=1.5/N;
    %settings=gammaSGVBsettings(6); 
    settings=gammaSGVBsettings(-1); 
    settings.plot=0;
    settings.testGrad=0;
    settings.stepSize=beta;
    settings.inita= priora*ones(D*K,1); 
    settings.initb= priorb*ones(D*K,1);
    settings.forgetting=1;
    settings.useAdadelta=0;
    tic(); 
    yy=y*y';
    [a,b,ll]=gammaSGVB_prox(@(w) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb),...
        D*K,settings); 
    gammaSGVBtime3(rep)=toc();
    trueCov=wtrue * wtrue' + sigma2*eye(D); 
    EW=reshape(a ./ b,[D K]); 
    gammaSGVBamari3(rep)=amariError(EW,wtrue);


    %settings=gammaSGVBsettings(6); 
    settings=gammaSGVBsettings(1); 
    settings.plot=0;
    settings.testGrad=0;
    settings.inita= priora*ones(D*K,1); 
    settings.initb= priorb*ones(D*K,1);
    settings.useAnalyticEntropy =1;
    settings.usePrior=1;
    %settings.forgetting=0.01;
    settings.forgetting=1.0;
    tic(); 
    yy=y*y';
    [a,b,ll]=gammaSGVB_sgd(@(w) gpfaLikelihood_prior(w,yy,N,K,D,sigma2,priora,priorb),...
        D*K,settings); 
    gammaSGVBtime1(rep)=toc();
    trueCov=wtrue * wtrue' + sigma2*eye(D); 
    EW=reshape(a ./ b,[D K]); 
    gammaSGVBamari1(rep)=amariError(EW,wtrue);


    %settings=gammaSGVBsettings(6); 
    settings=gammaSGVBsettings(1); 
    settings.plot=0;
    settings.testGrad=0;
    settings.inita= priora*ones(D*K,1); 
    settings.initb= priorb*ones(D*K,1);
    settings.useAnalyticEntropy =1;
    settings.usePrior=0;
    %settings.forgetting=0.01;
    settings.forgetting=1.0;
    tic(); 
    yy=y*y';
    [a,b,ll]=gammaSGVB_sgd(@(w) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb),...
        D*K,settings); 
    gammaSGVBtime2(rep)=toc();
    trueCov=wtrue * wtrue' + sigma2*eye(D); 
    EW=reshape(a ./ b,[D K]); 
    gammaSGVBamari2(rep)=amariError(EW,wtrue);



end

save pnmfResults.mat ss wtrue   ...
     gammaSGVBamari1 gammaSGVBtime1 gammaSGVBamari2 gammaSGVBtime2 ...
     gammaSGVBamari3 gammaSGVBtime3

%gammaSGVBamari gammaSGVBtime gammaSGVBPROXamari gammaSGVBPROXtime
%load pnmfResults

subplot(1,2,1); 
hold off
semilogx(ss,gammaSGVBamari1,'x--r','DisplayName','1');
xlabel('# data points N'); ylabel('Amari error');
hold on
semilogx(ss,gammaSGVBamari2,'+:b','DisplayName','2');
hold on
semilogx(ss,gammaSGVBamari3,'o-g','DisplayName','3');

legend('show','Location','northwest')
set(gca,'fontsize',16);

subplot(1,2,2); 
%for rep=1:length(ss)
    %nsfaTime(rep)=nsfaResTable{rep}(end,1); 
%end
hold off
loglog(ss,gammaSGVBtime1,'x--r','DisplayName','1');
xlabel('# data points N'); ylabel('run time (seconds)');
hold on
loglog(ss,gammaSGVBtime2,'+:b','DisplayName','2'); 
loglog(ss,gammaSGVBtime3,'o-g','DisplayName','3'); 
legend('show','Location','northwest')
set(gca,'fontsize',16);

%exportpdf('seminmf.pdf'); 
