function [a,b,sgvbCov,settings]=mygpfa(y,K, method, likelihood, sigma2, settings)

    D=size(y,1);
    N=size(y,2);
    
    yy=y*y';

    like = @(w) likelihood(w,yy,N);
    [a,b,ll,settings]=method(like,D*K,settings); 

    EW=reshape(a ./ b ,[D K]);
    sgvbCov=EW * EW' + eye(D) * sigma2;
    %{
    ecov=zeros(D,D); 
    nsamples=100;
    for i=1:nsamples
        W= gamrnd(a.W , 1./b.W );
        sigma2 = 1 / gamrnd(a.noisePrec, 1 ./ b.noisePrec);
        ecov=ecov+ W * W' + eye(D) * sigma2;
    end
    sgvbCov=ecov/nsamples; 
    %}
end
