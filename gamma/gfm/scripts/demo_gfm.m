is_full = 1;

load('./datasets/cytof.mat');

%out_path='results';
out_path='.';
y=bsxfun(@minus,y,mean(y,2));
y=bsxfun(@rdivide,y,std(y,[],2));

allres=table();

%intrain=[100, 200, 300, 500, 1000, 2000, 3000, 5000, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5]; 
intrain=[3e5]; 


clear res
K=39;
%K=35
D=39;

sigma2=.1;
priora=1.0;
priorb=1.0;
%nSamples=200;
nSamples=50;
maxIter=600;
decay = 0.8;

global time_cache;
global is_cache;
is_cache = is_full;

times_cache=zeros(maxIter+1,5);

for i=intrain
    ytrain=y(:,1:i);
    ytest=y(:,(i+1):end);

    method1 = @(like,dim,conf) gammaSGVB_rgvi(like,dim,conf);

    method2 = @(like,dim,conf) gammaSGVB_cvi_new(like,dim,conf);
    method3 = @(like,dim,conf) gammaSGVB_cvi(like,dim,conf);

    method4 = @(like,dim,conf) gammaSGVB_bbvi_new(like,dim,conf);
    method5 = @(like,dim,conf) gammaSGVB_bbvi(like,dim,conf);


    %%%%%%%%%%%%%%%%%
    %rgvi new
    %%%%%%%%%%%%%%%%%
    time_cache=zeros(maxIter+1,1);
    settings=gammaSGVBsettings(-1);
    settings.plot=0;
    settings.testGrad=0;
    settings.stepSize=0.00012;%K=39
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.samples=maxIter;
    settings.nSamples=nSamples;

    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    likelihood = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method1, likelihood, sigma2, settings);
    res.train1=settings.train;
    res.test1=settings.test;
    res.it1=settings.it;
    assert(time_cache(1) == 0)
    times_cache(:,1)=time_cache;


%{
    %%%%%%%%%%%%%%%%%
    %cvi new
    %%%%%%%%%%%%%%%%%
    time_cache=zeros(maxIter+1,1);
    settings=gammaSGVBsettings(-1);
    settings.decay = decay;
    settings.plot=0;
    settings.testGrad=0;
    settings.stepSize=0.00012;%K=39
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.samples=maxIter;
    settings.nSamples=nSamples;

    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    likelihood = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method2, likelihood, sigma2, settings);
    res.train2=settings.train;
    res.test2=settings.test;
    res.it2=settings.it;
    assert(time_cache(1) == 0)
    times_cache(:,2)=time_cache;
%}


    %%%%%%%%%%%%%%%%%
    %cvi
    %%%%%%%%%%%%%%%%%
    time_cache=zeros(maxIter+1,1);
    settings=gammaSGVBsettings(-1);
    settings.decay = decay;
    settings.plot=0;
    settings.testGrad=0;
    settings.stepSize=0.00012;%K=39
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.samples=maxIter;
    settings.nSamples=nSamples;

    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    likelihood = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method3, likelihood, sigma2, settings);
    res.train2=settings.train;
    res.test2=settings.test;
    res.it2=settings.it;
    assert(time_cache(1) == 0)
    times_cache(:,2)=time_cache;


%{
    %%%%%%%%%%%%%%%%%
    %adam new with exact entropy and prior
    %%%%%%%%%%%%%%%%%

    time_cache=zeros(maxIter+1,1);
    settings=gammaSGVBsettings(9);
    settings.plot=0;
    settings.testGrad=0;
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.usePrior=0;
    settings.stepSize=1.0;%1.2
    settings.samples=maxIter;
    settings.nSamples=nSamples;


    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    likelihood = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method4, likelihood, sigma2,  settings);
    res.train4=settings.train;
    res.test4=settings.test;
    res.it4=settings.it;
    assert(time_cache(1) == 0)
    times_cache(:,4)=time_cache;
%}


    %%%%%%%%%%%%%%%%%
    %adam with exact entropy and prior
    %%%%%%%%%%%%%%%%%

    time_cache=zeros(maxIter+1,1);
    settings=gammaSGVBsettings(9);
    settings.plot=0;
    settings.testGrad=0;
    settings.inita= priora*ones(D*K,1);
    settings.initb= priorb*ones(D*K,1);
    settings.usePrior=0;
    settings.stepSize=1.0;%1.2
    settings.samples=maxIter;
    settings.nSamples=nSamples;


    settings.D=D;
    settings.K=K;
    settings.ytrain=ytrain;
    settings.ytest=ytest;
    settings.sigma2=sigma2;

    likelihood = @(w,yy,N) gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,priora,priorb);
    [a,b,sgvbCov,settings]=mygpfa(ytrain, K, method5, likelihood, sigma2,  settings);
    res.train3=settings.train;
    res.test3=settings.test;
    res.it3=settings.it;
    assert(time_cache(1) == 0)
    times_cache(:,3)=time_cache;

    allres=[allres; struct2table(res)];
end

%algos={'rgvi-new', 'cvi-new', 'cvi','Adam-new', 'Adam'};
algos={'rgvi', 'cvi', 'Adam'};
file_name=sprintf('%s/cytof_%d_loss.mat',out_path,K)
save(file_name,'allres', 'algos')


num_method=3;
color_choice=['r','k','b','g','c','m','y'];
marker_choice=['s','d','x','+','*','o','^'];
figure(1);
clf;
plots=[];
for ii = 1:num_method
	switch ii
	case 1
	xp=allres.it1; yp=-allres.test1;
	case 2
	xp=allres.it2; yp=-allres.test2;
	case 3
	xp=allres.it3; yp=-allres.test3;
	case 4
	xp=allres.it4; yp=-allres.test4;
	case 5
	xp=allres.it5; yp=-allres.test5;
	otherwise
	error('do not support')
	end
        plot(xp,yp, '-', 'color', color_choice(ii), 'linewidth', 3);
        hold on;
        pl=plot(xp(1),yp(1), 'o', 'color', color_choice(ii), 'linewidth', 3, 'marker', marker_choice(ii), 'markerEdgecolor', color_choice(ii), 'markerFaceColor', [1 1 1]);
        plots(ii)=pl;

end
hold off;
ylim([40,120]);
hx = xlabel('# pass');
hy = ylabel('negative mean test log likelihood');
legend([plots], algos);
name=sprintf('GAMMA_%d-decay%.2f.pdf',K,decay);
print(name, '-dpdf')
