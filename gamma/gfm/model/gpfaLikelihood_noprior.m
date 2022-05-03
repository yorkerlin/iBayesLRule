function [l,g]=gpfaLikelihood_noprior(w,yy,N,K,D,sigma2,wshape,wrate)
    W=reshape(w,[D K]); 
    covar=W*W'+sigma2*eye(D); 
    ch=chol(covar); % prec=ch'*ch
    prec=ch\(ch'\eye(D)); 
%     er=ch\y; 
    l=-N*sum(log(diag(ch)))-.5*sum( yy(:) .* prec(:) )-N*D/2*log(2*pi); 
%     assert( sum(er(:).^2) == sum(diag(y'*(covar\y)))); 
%     assert( sum(er(:).^2) == sum( yy(:) .* prec(:) )); 
    if nargout()>1
	    %dl/dW=g= - [(K^{-1} - K^{-1}yy'K^{-1})W] 
      precW=ch\(ch'\W); 
      g=-N*precW + ch\(ch'\(yy*precW)); 
      g=g(:); 
    end
end
