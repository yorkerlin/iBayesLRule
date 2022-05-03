% range of sample sizes
ss=[10 20 30 40 50 100 200 500 1000 2000 5000 10000 ]; 

D=50;
K=10; 
wtrue=(rand(D,K)<.2) .*  rand(D,K); %abs(randn(D,K)); gamrnd(1,1,D,K);

%parfor rep=1:length(ss)
for rep=1:length(ss)

    N=ss(rep); 
    x=randn(K,N); 
    
    sigma2=.1; 
    y=wtrue * x + sqrt(sigma2) * randn(D,N); 

    % gammaSGVB GPFA ---------------
    settings=gammaSGVBsettings(6); 
    settings.plot=0;
    settings.testGrad=0;
    tic(); 
    yy=y*y';
    [a,b,ll]=gammaSGVB(@(w) gpfaLikelihood(w,yy,N,K,D,sigma2,.1,1),...
        D*K,settings); 
    gammaSGVBtime(rep)=toc();
    trueCov=wtrue * wtrue' + sigma2*eye(D); 
    EW=reshape(a ./ b,[D K]); 
    gammaSGVBamari(rep)=amariError(EW,wtrue);

    % SPCA ---------------
    temp=[];
    tic
    for i=1:20 % optimize over number of latent factors
        b=spca(y', [], K, Inf, -i);
        if size(b,2)<K
            b=[b .01*ones(D,K-size(b,2))];
        end
        temp(i)=amariError(b,wtrue);
    end
    spcaTime(rep)=toc
    spcaAmari(rep)=min(temp);
    
    % NSFA MCMC code ---------------
    settings=defaultsettings();
    [settings.D,settings.N]=size(y); 
    settings.iterations=1100;
    settings.verbose=0;
    settings.K=10; 
    settings.np=0; 
    settings.lambdae=1/sigma2; 
    settings.store_samples=100;
    settings.alpha=1;
    settings.output=['nsfa_res' num2str(rep) '.mat'];
    mvmask=ones(settings.D,settings.N);
    initialsample=initisFA(settings);
    [finalsample,resultstable]=isFA(y,mvmask,initialsample,settings);
    %plot(resultstable(:,1),resultstable(:,2),'k+-'); 
    %xlabel('cpu time');
    %ylabel('log joint probability');
    res=load(settings.output); 
    G=res.samples{1}.G; 
    isfaCov=G*G'; 
    for i=2:length(res.samples)
        G=G+res.samples{i}.G; 
        isfaCov=isfaCov+res.samples{i}.G*res.samples{i}.G'; 
    end
    G=G/length(res.samples); 
    nsfaAmari(rep)=amariError(G,wtrue);
    nsfaResTable{rep}=resultstable;     
end

save pnmfResults.mat ss wtrue   ...
     gammaSGVBamari gammaSGVBtime spcaAmari spcaTime nsfaAmari nsfaResTable 

%load pnmfResults

subplot(1,2,1); 
hold off
semilogx(ss,nsfaAmari,'x--r','DisplayName','NSFA'); xlabel('# data points N'); ylabel('Amari error');
hold on
semilogx(ss,gammaSGVBamari,'+:b','DisplayName','GPFA SGVB');
semilogx(ss,spcaAmari,'o-g','DisplayName','SPCA'); 
legend('show','Location','northwest')
set(gca,'fontsize',16);

subplot(1,2,2); 
for rep=1:length(ss)
    nsfaTime(rep)=nsfaResTable{rep}(end,1); 
end
hold off
loglog(ss,nsfaTime,'x--r','DisplayName','NSFA');
xlabel('# data points N'); ylabel('run time (seconds)');
hold on
loglog(ss,gammaSGVBtime,'+:b','DisplayName','GPFA SGVB'); 
loglog(ss,spcaTime,'o-g','DisplayName','SPCA');
legend('show','Location','northwest')
set(gca,'fontsize',16);

exportpdf('seminmf.pdf'); 
