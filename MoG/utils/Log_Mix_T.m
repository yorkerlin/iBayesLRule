function [varargout] = Log_Mix_T(xSamples,logMixWeights,mixMeans,mixPrecs,alpha)

loglikelihood =@(xSampled, Mean, Prec)log_T(xSampled, Mean, Prec, alpha);
[logConPrb, logConWt, logMarPrb] = Get_Conditional_Log_Prbsv2(xSamples,logMixWeights,mixMeans,mixPrecs,loglikelihood);

assert( size(logMixWeights, 2) == length(mixMeans) )
[d,nrSamples] = size(xSamples);

lp = logMarPrb'; %  \log p(z)
assert(size(lp,1) == nrSamples);

k = size(xSamples,1);
Precs = reshape(cell2mat(mixPrecs'),k,k,[]);
means = reshape(cell2mat(mixMeans'),k,[]);

if nargout>1
    %{
    g = zeros(d,nrSamples);
    if nargout>2
        H = zeros(d,d,nrSamples);
    end
    for m=1:nrSamples
        gn = 0;
        % gradient
        for j=1:length(mixMeans)
            %p(w) * p(z|w) / p(z) = p (w|z) = logConWt
            cholPrec = chol(mixPrecs{j});
            half = cholPrec*(mixMeans{j}-xSamples(:,m));
            tmp_com = 1.0+( sum(half.^2) )/(2.0*alpha);
            factor1 = (alpha+d/2.)/(alpha*tmp_com);
            gn = gn + weightProd(logConWt(j,m),mixPrecs{j}*(mixMeans{j}-xSamples(:,m))*factor1);
        end
        g(:,m) = gn; %  \nabla_z \log p(z)

        if nargout>2
            Hn = 0;
            % hessian
            for j=1:length(mixMeans)
                cholPrec = chol(mixPrecs{j});
                half = cholPrec*(mixMeans{j}-xSamples(:,m));
                tmp_com = 1.0+( sum(half.^2) )/(2.0*alpha);
                factor1 = (alpha+d/2.)/(alpha*tmp_com);
                factor2 = (alpha+d/2.+1)/(alpha*tmp_com);
                Hn = Hn - weightProd(logConWt(j,m),mixPrecs{j}*factor1) + ...
                weightProd(logConWt(j,m),(mixPrecs{j}*(mixMeans{j}-xSamples(:,m))*factor2-gn)*(mixPrecs{j}*(mixMeans{j}-xSamples(:,m))*factor1)');
            end
            H(:,:,m) = Hn; % \nabla_z^2 \log p(z)
        end

    end
    %}

    tmp0 = reshape(means-reshape(xSamples,k,1,[]),k,1,[],nrSamples);
    tmp1 = mtimesx(Precs,tmp0);
    tmp_com = 1.0+sum(tmp1 .* tmp0,1)/(2.0*alpha);
    factor1 = (alpha+d/2.)./(alpha*tmp_com);
    factor2 = (alpha+d/2.+1)./(alpha*tmp_com);

    g = sum( reshape(weightProd2(reshape(logConWt,[],1),reshape(tmp1.*factor1,k,[])),k,[],nrSamples), 2);
    g = reshape(g,k,nrSamples);
    if nargout>2
        tmp2 = mtimesx(reshape(reshape(tmp1.*factor2,k,[],nrSamples)-reshape(g,k,1,[]),k,1,[]),reshape(tmp1.*factor1,1,k,[]));
        H = -sum(reshape(weightProd2(reshape(logConWt,[],1),reshape(repmat(Precs,1,1,1,nrSamples).*factor1-reshape(tmp2,k,k,[],nrSamples),k,k,[])),k,k,[],nrSamples),3);
        H = reshape(H,k,k,nrSamples);
        H = (H+permute(H,[2,1,3]))/2;
    end
end

if nargout==1
    varargout = {lp};
else
    if nargout>1
        varargout = {lp, g};
        if nargout>2
            varargout = {lp, g, H};
        end
    end
end

end
