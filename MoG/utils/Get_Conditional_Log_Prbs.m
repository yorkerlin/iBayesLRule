function [logConPrb, logConWt, logMarPrb] = Get_Conditional_Log_Prbs(xSampled,logMixWeights,mixMeans,mixPrecs)
%ConPrb q(z|w)
%ConWt  q(w|z)
%MarPrb q(z)


[k, nrSamples] = size(xSampled);
nrComponents = length(mixMeans);
assert( length(logMixWeights) == nrComponents )

% get densities at the sampled points
logConPrb = zeros(nrComponents,nrSamples);
for c=1:nrComponents
    assert(logMixWeights(c) <= 0)
    assert( k == length(mixMeans{c}) );
    logConPrb(c,:) = log_gauss(xSampled, mixMeans{c}, mixPrecs{c});
end

logConWt = logConPrb + repmat(logMixWeights',1,nrSamples);
max_log_v = max(logConWt, [], 1);
logMarPrb = log1p(sum( exp(logConWt - max_log_v) ,1) -1) + max_log_v;
logConWt = logConWt - repmat(logMarPrb,nrComponents,1);

end

function lp = log_gauss(xSampled, Mean, Prec)
[k, nrSamples] = size(xSampled);
cholPrec = chol(Prec);
lp = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec))) ...
    -0.5*sum((cholPrec*(xSampled-repmat(Mean,1,nrSamples))).^2,1));
end

