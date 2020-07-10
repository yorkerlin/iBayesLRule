function [grad, hess] = InformativeGradFun(theta, y, sigma, wt)
if nargin<4
    wt = 1.0;
end

    prior_grad = -theta*wt;
    nonlinear_grad = [ (2*(theta(1).^3)-theta(1))*6.0;  2.0*theta(2) ];

    s2 = sigma*sigma/wt;
    likelihood_grad = -( nonlinear(theta)-y )/s2 * nonlinear_grad;

    grad = prior_grad + reshape( likelihood_grad, size(theta) );

if nargout>1
    prior_hess = -eye(length(theta))*wt;
    nonlinear_hess = [ (6*theta(1)).^2 - 6.0, 0;0, 2];
    likelihood_hess = -( nonlinear_grad*nonlinear_grad' + (nonlinear(theta)-y )*nonlinear_hess )/s2;
    hess = prior_hess + likelihood_hess;
end

end

function fv = nonlinear(theta)
    tsq2 = theta.^2;
    fv = (tsq2(1)-1)*tsq2(1)*3.0 + tsq2(2);
end




%theta = ones(2,1)/2;
%y = 1.0;
%sigma = 0.5;
%[grad, hess] = InformativeGradFun(theta, y, sigma)
