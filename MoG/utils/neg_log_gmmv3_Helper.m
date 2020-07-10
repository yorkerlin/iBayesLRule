function [grb,tmp1,hrb] = neg_log_gmmv3_Helper(xSampled,logRBindicator,means,Precs);
[k, nrSamples] = size(xSampled);

%tmp1 = einsum('ijc,jcm->icm', Precs, means - reshape(xSampled,k,1,[]));
tmp1 = reshape(mtimesx(Precs,reshape(means-reshape(xSampled,k,1,[]),k,1,[],nrSamples)), k, [], nrSamples);
gn = sum(reshape(weightProd2(reshape(logRBindicator,[],1), reshape(tmp1,k,[])), k, [], nrSamples),2);
% grb = - \nabla_z log q(z)
grb = -reshape(gn,k,[]); %k by nrSamples
if nargout>2
    %tmp2 = reshape(Precs-einsum('icm,jcm->ijcm', tmp1-gn,tmp1), k,k, []);
    tmp2 = reshape(Precs-mtimesx(reshape(tmp1-gn,k,1,[],nrSamples),reshape(tmp1,1,k,[],nrSamples)), k,k,[]);
    %hrb = - \nabla_z^2 log q(z)
    hrb = sum( reshape(weightProd2(reshape(logRBindicator,[],1), tmp2), k,k,[],nrSamples),3);
end
end
