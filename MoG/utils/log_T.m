function lp = log_T(xSampled, Mean, Prec, alpha)
assert(alpha>0)
[k, nrSamples] = size(xSampled);
cholPrec = chol(Prec);
lp = ( gammaln(alpha+k/2) - gammaln(alpha)-(k/2)*log(2*pi*alpha)+sum(log(diag(cholPrec))) ...
-(alpha+k/2)*log1p(sum((cholPrec*(xSampled-repmat(Mean,1,nrSamples))).^2,1)/(2*alpha)) );
end
