function comps = SampleMixture_comps(llp, nrSamples, rawUnif)
%generate the mixture index according to the log probability weigths (llp)

n=length(llp);
assert(size(llp,1) == 1);
assert(size(llp,2) == n);

if n==1
    comps=ones(nrSamples,1);
else
    assert( all(size(rawUnif) == [nrSamples,1] ) );

    [nllp,ind]=sort(llp,'descend');
    cnllp = zeros(n,1);
    cnllp(1) = nllp(1);
    logsum=nllp(1);
    for i=2:n
        cnllp(i)=log1p(exp(nllp(i)-logsum))+logsum;
        logsum=cnllp(i);
    end
    cpb = (cnllp-logsum)';
    ss=1+sum(repmat(log(rawUnif),1,n)>repmat(cpb,nrSamples,1),2);
    comps = arrayfun(@(x) ind(x), ss);
end
end
