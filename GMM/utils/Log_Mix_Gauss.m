function [varargout] = Log_Mix_Gauss(xSamples,logMixWeights,mixMeans,mixPrecs)

[logConPrb, logConWt, logMarPrb] = Get_Conditional_Log_Prbs(xSamples,logMixWeights,mixMeans,mixPrecs);

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
        %gn = 0;
        %% gradient
        %for j=1:length(mixMeans)
            %%p(w) * p(z|w) / p(z) = p (w|z) = logConWt
            %gn = gn + weightProd(logConWt(j,m),mixPrecs{j}*(mixMeans{j}-xSamples(:,m)));
        %end
        %%tmp1 = einsum('ijc,jc->ic', Precs, means-xSamples(:,m));
        tmp1 = reshape( mtimesx(Precs,reshape(means-xSamples(:,m),k,1,[])), k, []);
        gn = sum(weightProd2(logConWt(:,m), tmp1),2);
        g(:,m) = gn; %  \nabla_z \log p(z)

        if nargout>2
            %Hn = 0;
            %% hessian
            %for j=1:length(mixMeans)
                %Hn = Hn - weightProd(logConWt(j,m),mixPrecs{j}) + weightProd(logConWt(j,m),(mixPrecs{j}*(mixMeans{j}-xSamples(:,m))-gn)*(mixPrecs{j}*(mixMeans{j}-xSamples(:,m)))');
            %end
            %%Hn = -sum(weightProd2(logConWt(:,m),Precs-einsum('ic,jc->ijc', tmp1-gn,tmp1)),3);
            Hn = -sum(weightProd2(logConWt(:,m),Precs-mtimesx(reshape(tmp1-gn,k,1,[]),reshape(tmp1,1,k,[]))),3);
            H(:,:,m) = Hn; % \nabla_z^2 \log p(z)
        end

    end
    %}
    tmp1 = mtimesx(Precs,reshape(means-reshape(xSamples,k,1,[]),k,1,[],nrSamples));
    g = sum( reshape(weightProd2(reshape(logConWt,[],1),reshape(tmp1,k,[])),k,[],nrSamples), 2);
    g = reshape(g,k,nrSamples);
    if nargout>2
        tmp2 = mtimesx(reshape(reshape(tmp1,k,[],nrSamples)-reshape(g,k,1,[]),k,1,[]),reshape(tmp1,1,k,[]));
        H = -sum(reshape(weightProd2(reshape(logConWt,[],1),reshape(Precs-reshape(tmp2,k,k,[],nrSamples),k,k,[])),k,k,[],nrSamples),3);
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
