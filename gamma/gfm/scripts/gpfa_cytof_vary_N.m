load('data/cytof.mat');

y=bsxfun(@minus,y,mean(y,2));
y=bsxfun(@rdivide,y,std(y,[],2));

allres=table();

intrain=[100, 200, 300, 500, 1000, 2000, 3000, 5000, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5]; 

settings=gammaSGVBsettings(6); 
settings.plot=0;
settings.testGrad=0;
settings.samples=5000;

clear res
for i=intrain
    
    ytrain=y(:,1:i);
    ytest=y(:,(i+1):end); 

    K=39;
    [a,b,sgvbCov]=gpfaSigma(ytrain,K,settings);
    res.trainSnmf=mean(mvnpdfl(ytrain', zeros(1,D), sgvbCov));
    res.testSnmf=mean(mvnpdfl(ytest', zeros(1,D), sgvbCov));
    
    lwcov=shrinkcov(ytrain');
    res.trainlw=mean(mvnpdfl(ytrain', zeros(1,D), lwcov));
    res.testlw=mean(mvnpdfl(ytest', zeros(1,D), lwcov));

    empCov=cov(ytrain'); 
    res.trainEmp=mean(mvnpdfl(ytrain', zeros(1,D), empCov));
    res.testEmp=mean(mvnpdfl(ytest', zeros(1,D), empCov));

    allres=[allres; struct2table(res)];
end
hold off
loglog( intrain, -allres.testEmp, 'x--');
hold on
loglog( intrain, -allres.testlw, '+-');
loglog( intrain, -allres.testSnmf, 'o:');
legend('MLE','Ledoit-Wolfe','GPFA'); 
xlabel('# training samples'); 
ylabel('negative mean test log likelihood'); 
set(gca,'fontsize', 12);