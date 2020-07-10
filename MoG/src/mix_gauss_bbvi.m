function [mixWeights,mixMeans,mixPrecs] = mix_gauss_bbvi(option, likelihoodFun,nrComponents,nrSteps,init_m,init_P,preIt,nrSamples,stepSize,decay_mix,dataset_name,callbackFun,init_post)

%adam parameters
epsilon=1e-8; decay_factor_mean=0.9; decay_factor_var=0.999;

if nargin>12
    file_name = sprintf('%s-bbvi-%s-%d-%.8f-%.8f.mat',dataset_name,option,nrComponents,stepSize,decay_mix)
    callbackInfo = zeros(nrSteps,2);
end

% dimension
k = size(init_m,1);
xMode = init_m;
xNegHess = init_P;

lr2 = stepSize*decay_mix;
lr1 = stepSize;

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

tlam_mu = cell(nrComponents,1);
tlam_C = cell(nrComponents,1);

g_mean_mu = cell(nrComponents,1);
g_var_mu = cell(nrComponents,1);
g_mean_C = cell(nrComponents,1);
g_var_C = cell(nrComponents,1);

for c=1:nrComponents
    tlam_mu{c} = mixMeans{c};
    tlam_C{c} = chol(mixPrecs{c}\eye(k),'lower');

    g_mean_mu{c} = zeros(k,1);
    g_mean_C{c} = zeros(k,k);

    g_var_mu{c} = zeros(k,1);
    g_var_C{c} = zeros(k,k);
end

if nrComponents>1
    tlam_log_pi = log( mixWeights(1,1:end-1) ) - log( mixWeights(1,end) );
    g_mean_log_pi = zeros(1,nrComponents-1);
    g_var_log_pi = zeros(1,nrComponents-1);
else
    tlam_log_pi = 0;
end
logMixWeights = log(mixWeights);

% do stochastic approximation
for i=1:(nrSteps)

    if nrComponents>1 %\pi_i = exp(eta_i) / (1+sum(exp(eta))), i=1,...,(k-1)
        max_tlam_w = max(max(tlam_log_pi),0);
        norm_log_w = log1p( sum( exp(tlam_log_pi - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
        logMixWeights(1,1:end-1) = tlam_log_pi - norm_log_w;
        logMixWeights(1,end) = -norm_log_w;
    end

    if i==1 || mod(i,preIt) == 0
        exp( logMixWeights )
    end

    [xSampled,logSampDensPerComp] = SampleFromMixture(logMixWeights,mixMeans,mixPrecs,nrSamples);
    %xSampled  ~ q(z)
    %SampDensPerComp = q(z|w)

    [logRBindicator,logTotalSampDens] = CombineMixtureComponents(logMixWeights,logSampDensPerComp);
    %RBindicator  q(w|z)
    %TotalSampDens q(z)

    %compute log likelihood with prior  log p(z,x)
    switch option
    case {'hess'}
        [lpDens,grad,hess] = likelihoodFun(xSampled,i);
        hess_lik = hess;
    case {'first'}
        [lpDens,grad] = likelihoodFun(xSampled,i);
    end
    grad_lik = grad;
    %grad = num2cell(grad, [1])';
    %hess = squeeze( num2cell(hess,[1,2]) );

    %maxLogRB=max(logRBindicator,[],2);
    %log_sxxWeights = log1p(sum(exp(logRBindicator-maxLogRB),2)-1) + maxLogRB-log(size(logRBindicator,2));
    log_sxxWeights = logMixWeights';

    % log p(x) - log q(x)
    %sxyWeights = matrixProd(logRBindicator,(lpDens-logTotalSampDens')/nrSamples);
    [log_sxyWeights sig]= logMatrixProdv2(logRBindicator,(lpDens-logTotalSampDens')/nrSamples);

    %compute gradient w.r.t. log \pi
    g_log_pi = exp(log_sxyWeights - log_sxxWeights + logMixWeights') .* sig;
    g_log_pi(sig==0) = 0;
    g_log_pi= g_log_pi';

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

    g_mu = gmu;
    for c=1:nrComponents
        [gmu,g_mean_mu_tmp,g_var_mu_tmp] = adam(-g_mu(:,c),g_mean_mu{c},g_var_mu{c},lr1,i,decay_factor_mean,decay_factor_var,epsilon);
        g_mean_mu{c} = g_mean_mu_tmp;
        g_var_mu{c} = g_var_mu_tmp;
        tlam_mu{c} = tlam_mu{c} - gmu;

        g_C_c = tril(gV(:,:,c)*(2*tlam_C{c})); % V= CC'
        [gC,g_mean_C_tmp,g_var_C_tmp] = adam(-g_C_c,g_mean_C{c},g_var_C{c},lr1,i,decay_factor_mean,decay_factor_var,epsilon);
        g_mean_C{c} = g_mean_C_tmp;
        g_var_C{c} = g_var_C_tmp;
        tlam_C{c} = tlam_C{c} - gC;

        mixPrecs{c} = tlam_C{c}'\(tlam_C{c}\eye(k));
        mixMeans{c} = tlam_mu{c};
    end

    if nrComponents>1
        %gradient w.r.t eta using the chain rule
        %g_eta_i = g_log_pi_i - sum(g_log_pi)*pi_i

        %note:
        %log pi_i = eta_i - log( 1+ sum( exp(eta) ) when 1<=i<=K-1
        %log pi_K = - log( 1+ sum( exp(eta) ) when i==K
        %by the chain rule, we have
        %grad_{eta_i} = grad_{log_pi_i} - \sum_{c=1}^{K} grad_{pi_c}*exp(eta_i)/(1+ sum( exp(eta) ) )
        %             = grad_{log_pi_i} - \sum_{c=1}^{K} grad_{pi_c}*pi_i
        g_pi = g_log_pi(1,1:end-1) - sum(g_log_pi).*exp(logMixWeights(1,1:end-1));
        [glog_pi,g_mean_log_pi,g_var_log_pi] = adam(-g_pi,g_mean_log_pi,g_var_log_pi,lr2,i,decay_factor_mean,decay_factor_var,epsilon);
        assert( all( size(tlam_log_pi) == size(glog_pi) ) )
        tlam_log_pi = tlam_log_pi - glog_pi; % It should be eta in math. I use log_pi to denote eta.
    end

    if nargin>11 && ( (i<=1e5 && i>=1e3 && mod(i,200)== 0) ||  (i<1000 && mod(i,50)== 0) || (i<100 && mod(i,10)== 0) || mod(i,preIt) == 0)
    %if nargin>11 && ( mod(i,preIt) == 0 || i==1 )

        if nrComponents>1
            max_tlam_w = max(max(tlam_log_pi),0);
            norm_log_w = log1p( sum( exp(tlam_log_pi - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
            mixWeights(1,1:end-1) = exp(tlam_log_pi - norm_log_w);
            mixWeights(1,end) = 1-sum(mixWeights(1,1:end-1));
        end

        post_dist.mixWeights=mixWeights;
        post_dist.mixMeans=mixMeans;
        post_dist.mixPrecs=mixPrecs;
        post_dist.tlam_w=tlam_log_pi;

        if nargin>12
            callbackInfo(i,:) = callbackFun(i,post_dist);
        else
            callbackFun(i,post_dist);
        end

    end

end

if nrComponents>1
    max_tlam_w = max(max(tlam_log_pi),0);
    norm_log_w = log1p( sum( exp(tlam_log_pi - max_tlam_w) )-1+ exp(-max_tlam_w) ) + max_tlam_w;
    mixWeights(1,1:end-1) = exp(tlam_log_pi - norm_log_w);
    mixWeights(1,end) = 1-sum(mixWeights(1,1:end-1));
end

if nargin>12
    save(file_name,'preIt','nrSamples','stepSize','decay_mix','nrComponents','nrSteps','init_m','init_P','callbackInfo');
end

end
