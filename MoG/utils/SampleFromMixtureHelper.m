% sample from GMM
function [xSampled,logSampDensPerComp,cholPrec] = SampleFromMixtureHelper(logMixWeights,mixMeans,mixPrecs,rawNorm,rawUnif)

nrSamples = size(rawUnif,1);

% dimensions
k = length(mixMeans{1});
nrComponents = length(mixMeans);

% get cholesky's of the precision matrices of all Gaussian components
cholPrec = cell(nrComponents,1);
for c=1:nrComponents
    cholPrec{c} = chol(mixPrecs{c});
end

% sample mixture component indicators
comps = SampleMixture_comps(logMixWeights, nrSamples, rawUnif);
assert( all(size(rawNorm) == [k, nrSamples]) );

% sample from the Gaussian mixture components
xSampled = zeros(k,nrSamples);
z = rawNorm;

for j=1:nrSamples
    xSampled(:,j) = mixMeans{comps(j)} + cholPrec{comps(j)}\z(:,j);
end

% get densities at the sampled points
logSampDensPerComp = zeros(nrComponents,nrSamples);
for c=1:nrComponents
    if k>1
    logSampDensPerComp(c,:) = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*sum((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
    else
    logSampDensPerComp(c,:) = (-(k/2)*log(2*pi)+sum(log(diag(cholPrec{c}))) ...
        -0.5*((cholPrec{c}*(xSampled-repmat(mixMeans{c},1,nrSamples))).^2));
    end
end
end


