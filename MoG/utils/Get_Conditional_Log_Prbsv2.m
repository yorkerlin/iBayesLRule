function [logConPrb, logConWt, logMarPrb] = Get_Conditional_Log_Prbsv2(xSampled,logMixWeights,mixMeans,mixPrecs,logCondLikelihood)
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
    logConPrb(c,:) = logCondLikelihood(xSampled, mixMeans{c}, mixPrecs{c});
end
logConWt = logConPrb + repmat(logMixWeights',1,nrSamples);
max_log_v = max(logConWt, [], 1);
logMarPrb = log1p(sum( exp(logConWt - max_log_v) ,1) -1) + max_log_v;
logConWt = logConWt - repmat(logMarPrb,nrComponents,1);

end
