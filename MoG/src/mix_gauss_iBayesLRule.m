function [mixWeights,mixMeans,mixPrecs] = mix_gauss_iBayesLRule(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P, preIt, nrSamples, stepSize, decay_mix, dataset_name,callbackFun, init_post)

if nargin>12
    file_name = sprintf('%s-iBayesLRule-%s-%d-%.8f-%.8f.mat',dataset_name,option,nrComponents,stepSize,decay_mix)
    callbackInfo = zeros(nrSteps,2);
end

% dimension
k = size(init_m,1);
xMode = init_m;
xNegHess = init_P;

% use maximum + curvature for initialization
cholNegHess = chol(xNegHess);
mixMeans = cell(nrComponents,1);
mixPrecs = cell(nrComponents,1);


for c=1:nrComponents
if nargin<13
    mixMeans{c} = xMode + (cholNegHess\randn(k,1));
    mixPrecs{c} = xNegHess;
else
    mixMeans{c} = init_post.mixMeans{c};
    mixPrecs{c} = init_post.mixPrecs{c};
end
end


mixWeights = ones(1,nrComponents)/nrComponents;

if nrComponents>1
    tlam_w = log( mixWeights(1,1:end-1) ) - log( mixWeights(1,end) );
    g_mean_m_w = zeros(1,nrComponents-1);
    g_var_m_w = zeros(1,nrComponents-1);
    %assert( all(size(tlam_w) == size( g_mean_m_w)) )
else
    tlam_w = 0;
end
logMixWeights = log(mixWeights);

% do stochastic approximation
for i=1:(nrSteps)
    if nrComponents>1
        max_tlam_w = max(max(tlam_w),0);
        norm_log_w = log1p( sum( exp(tlam_w - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
        logMixWeights(1,1:end-1) = tlam_w - norm_log_w;
        logMixWeights(1,end) = -norm_log_w;
    end

    if i==1 || mod(i,preIt) == 0
        exp( logMixWeights )
    end

    [xSampled,logSampDensPerComp,mixUpperCholPrecs] = SampleFromMixture(logMixWeights,mixMeans,mixPrecs,nrSamples);
    %xSampled = z ~ q(z)
    %logSampDensPerComp = log( q(z|w) )


    [logRBindicator,logTotalSampDens] = CombineMixtureComponents(logMixWeights,logSampDensPerComp);
    %logRBindicator = log( q(w|z) )
    %logTotalSampDens =log( q(z) )

    %compute log likelihood with prior  log p(z,x)

    switch option
    case {'hess'}
        [lpDens,grad,hess] = likelihoodFun(xSampled,i);
        hess_lik = hess;
    case {'first'}
        [lpDens,grad] = likelihoodFun(xSampled,i);
    end
    grad_lik = grad;

    log_sxxWeights = logMixWeights';%log( q(w) )

    mode = 1;
    if mode==1
        %sxyWeights = E_{q(z)} [ q(z|w)/q(z) (log p(z,x) - log q(z))  ]
        %           = E_{q(z)} [ q(w|z)/q(w) (log p(z,x) - log q(z))  ]
        [log_sxyWeights2 sig2]= logMatrixProdv3(logRBindicator-log_sxxWeights-log(nrSamples),lpDens-logTotalSampDens');
        g_m_w2 = exp(log_sxyWeights2) .* sig2;
    else
        [log_sxyWeights3 sig3]= logMatrixProdv3(logRBindicator-log_sxxWeights,lpDens-logTotalSampDens');
        g_m_w2 = exp(log_sxyWeights3) .* sig3;
        tmpp = mean(lpDens-logTotalSampDens')*sum(logRBindicator-log_sxxWeights,2);%C by 1
        assert( nrSamples > 1)
        g_m_w2 = (g_m_w2 - tmpp) / (nrSamples-1); %a better/worse estimator ?
    end

    g_m_w2= g_m_w2';

    Precs = reshape(cell2mat(mixPrecs'),k,k,[]);
    means = reshape(cell2mat(mixMeans'),k,[]);

    switch option
    case {'hess'}
        [grad_ent,hess_ent] = neg_log_gmmv3(xSampled,logRBindicator,means,Precs);
        lwt = logRBindicator - log_sxxWeights;
        tmp3 = reshape(permute(repmat(grad_lik+grad_ent,1,1,nrComponents),[1,3,2]),k,[]);
        g = reshape(weightProd2(reshape(lwt,[],1),tmp3),k,[],nrSamples);
        gmu = mean(g,3);

        tmp4 = reshape(permute(repmat(hess_lik+reshape(hess_ent,k,k,[]),1,1,1,nrComponents),[1,2,4,3]),k,k,[]);
        h = reshape(weightProd2(reshape(lwt,[],1),tmp4),k,k,[],nrSamples);
        gV = mean(h,4)/2;
    case {'first'}
        [grad_ent,tmp1,hess_ent] = neg_log_gmmv3_Helper(xSampled,logRBindicator,means,Precs);

        lwt = logRBindicator - log_sxxWeights;
        tmp3 = reshape(permute(repmat(grad_lik+grad_ent,1,1,nrComponents),[1,3,2]),k,[]);
        g = reshape(weightProd2(reshape(lwt,[],1),tmp3),k,[],nrSamples);
        gmu = mean(g,3);

        %tmp = einsum('icm,jcm->ijcm',-tmp1,permute(repmat(grad_lik,1,1,nrComponents), [1,3,2]));
        tmp = mtimesx(reshape(-tmp1,k,1,[], nrSamples),reshape(permute(repmat(grad_lik,1,1,nrComponents),[1,3,2]),1,k,[],nrSamples));
        gV = mean(reshape(weightProd2(reshape(lwt,[],1),reshape((tmp+permute(tmp,[2,1,3,4]))/2+ hess_ent, k,k,[])),k,k,[],nrSamples),4)/2;
    end

    lr1 = stepSize;

    for c=1:nrComponents
        %mixMeans{c} = mixMeans{c} + lr1*(mixPrecs{c}\gmu{c});
        %mixPrecs{c} = mixPrecs{c} - 2*lr1*gV{c}+ (lr1*lr1*2)*(gV{c}*(mixPrecs{c}\gV{c}));
        %mixMeans{c} = mixMeans{c} + lr1*(mixPrecs{c}\gmu(:,c));
        %mixPrecs{c} = mixPrecs{c} - 2*lr1*gV(:,:,c)+ (lr1*lr1*2)*(gV(:,:,c)*(mixPrecs{c}\gV(:,:,c)));
        L = mixUpperCholPrecs{c}';
        mixMeans{c} = mixMeans{c} + lr1*(L'\(L\gmu(:,c)));
        right = L\gV(:,:,c);
        R = L' - (lr1*2.0)*right;
        mixPrecs{c} = (mixPrecs{c} + R'*R)/2; %mixPrecs{c} = mixPrecs{c} - lr1*(2.0*gV(:,:,c)) + (lr1*lr1*2.0)*(right'*right);
    end

    lr2 = lr1*decay_mix;
    %beta1 = 1.0-1e-8;
    %decay1 = (1.0-beta1)/(1.0-power(beta1,i));
    %rwt = 5e5*decay1;%saliman's 2d

    if nrComponents>1
        g_m_w = g_m_w2(1,1:end-1) - g_m_w2(1,end);
        %assert( all(size(tlam_w) == size(g_m_w)) )
        tlam_w = tlam_w + lr2*g_m_w;
        %tlam_w = (1.0-rwt*lr2)*tlam_w + lr2*g_m_w; %entropy regularization on q(w) by adding rwt * E_q{w}[ -log q(w) ]
    end

    if nargin>11 && ( (i<=1e5 && i>=1e3 && mod(i,200)== 0) ||  (i<1000 && mod(i,50)== 0) || (i<100 && mod(i,10)== 0) || mod(i,preIt) == 0)
    %if nargin>11 && ( mod(i,preIt) == 0 || i==1 )

        if nrComponents>1
            max_tlam_w = max(max(tlam_w),0);
            norm_log_w = log1p( sum( exp(tlam_w - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
            mixWeights(1,1:end-1) = exp(tlam_w - norm_log_w);
            mixWeights(1,end) = 1-sum(mixWeights(1,1:end-1));
        end

        post_dist.mixWeights=mixWeights;
        post_dist.mixMeans=mixMeans;
        post_dist.mixPrecs=mixPrecs;
        post_dist.tlam_w=tlam_w;

        if nargin>12
            callbackInfo(i,:) = callbackFun(i,post_dist);
        else
            callbackFun(i,post_dist);
        end

    end
end

if nrComponents>1
    max_tlam_w = max(max(tlam_w),0);
    norm_log_w = log1p( sum( exp(tlam_w - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
    logMixWeights(1,1:end-1) = tlam_w - norm_log_w;
    logMixWeights(1,end) = -norm_log_w;
    mixWeights = exp( logMixWeights);
end

if nargin>12
    save(file_name,'preIt','nrSamples','stepSize','decay_mix','nrComponents','nrSteps','init_m','init_P','callbackInfo');
end

end

