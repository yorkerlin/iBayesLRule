load('../datasets/cytof.mat');

y=bsxfun(@minus,y,mean(y,2));
y=bsxfun(@rdivide,y,std(y,[],2));

allres=table();

%intrain=[100, 200, 300, 500, 1000, 2000, 3000, 5000, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5]; 
intrain=[3e5]; 


clear res
%K=39;
K=30
D=39;

sigma2=.1;
priora=1.0;
priorb=1.0;
nSamples=50;
maxIter=500;
for i=intrain
    ytrain=y(:,1:i);
    ytest=y(:,(i+1):end); 

    %{

    likelihood2 = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    %settings=gammaSGVBsettings(6); 
    settings=gammaSGVBsettings(-1);
    settings.plot=0;
    settings.testGrad=0;
    %settings.stepSize= 0.00010;%K=30
    settings.stepSize=0.00001;%K=30
    %settings.stepSize=0.00005;%K=39
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.forgetting=1;
    settings.useAdadelta=0;
    settings.samples=maxIter;
    settings.nSamples=nSamples;

    settings.mc='inexact_g+exact_I';
    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    method2 = @(like,dim,conf) gammaSGVB_prox2(like,dim,conf);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method2, likelihood2, sigma2, settings);
    res.train1=settings.train;
    res.test1=settings.test;
    res.it1=settings.it;

%}


    settings=gammaSGVBsettings(1);
    settings.plot=0;
    settings.testGrad=0;
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.usePrior=1;
    settings.stepSize=0.001;
    settings.samples=maxIter;
    settings.nSamples=nSamples;

    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    likelihood1 = @(w,yy,N) gpfaLikelihood_prior(w,yy,N,K,D,sigma2,priora,priorb);
    method1 = @(like,dim,conf) gammaSGVB_sgd2(like,dim,conf);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method1, likelihood1, sigma2, settings);
    settings
    res.train2=settings.train;
    res.test2=settings.test;
    res.it2=settings.it;


    likelihood2 = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    %settings=gammaSGVBsettings(6); 
    settings=gammaSGVBsettings(-1);
    settings.plot=0;
    settings.testGrad=0;
    settings.stepSize= 0.00010;%K=30
    %settings.stepSize=0.00001;%K=30
    %settings.stepSize=0.00005;%K=39
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.forgetting=1;
    settings.useAdadelta=0;
    settings.samples=maxIter;
    settings.nSamples=nSamples;

    settings.mc='inexact_g+inexact_I';
    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    method2 = @(like,dim,conf) gammaSGVB_prox2(like,dim,conf);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method2, likelihood2, sigma2, settings);
    res.train3=settings.train;
    res.test3=settings.test;
    res.it3=settings.it;


    allres=[allres; struct2table(res)];
end
hold off
%loglog( intrain, -allres.test1, 'x--');
loglog( allres.it1, -allres.test1, 'x--');
hold on
loglog( allres.it2, -allres.test2, '+-');
loglog( allres.it3, -allres.test3, 'o:');
legend('1','2','3'); 
xlabel('# pass'); 
ylabel('negative mean test log likelihood'); 
ylim([40,180]);
set(gca,'fontsize', 12);
%name=sprintf('GAMMA_%d.pdf',K);
%print(name, '-dpdf')
