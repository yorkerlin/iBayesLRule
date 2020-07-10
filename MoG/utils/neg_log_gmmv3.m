function [grb,hrb] = neg_log_gmmv3(xSampled,logRBindicator,means,Precs);
if nargout>1
    [grb,~,hrb] = neg_log_gmmv3_Helper(xSampled,logRBindicator,means,Precs);
else
    grb = neg_log_gmmv3_Helper(xSampled,logRBindicator,means,Precs);
end
end
