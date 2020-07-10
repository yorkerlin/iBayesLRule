% sample from the approximate posterior
function [xSampled,logSampDensPerComp,mixUpperCholPrecs] = SampleFromMixture(logMixWeights,mixMeans,mixPrecs,nrSamples)
k = length(mixMeans{1});
rawNorm = randn(k,nrSamples);
rawUnif = rand(nrSamples,1);
[xSampled,logSampDensPerComp,mixUpperCholPrecs] = SampleFromMixtureHelper(logMixWeights,mixMeans,mixPrecs,rawNorm,rawUnif);
end
