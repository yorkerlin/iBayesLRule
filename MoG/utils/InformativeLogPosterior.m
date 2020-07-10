function lp = InformativeLogPosterior(theta, y, sigma)
    prior = sum( log_uni_gauss(theta, 0.0, 1.0) );
    likelihood = log_uni_gauss(y, nonlinear(theta) , sigma);
    lp = prior + likelihood;
end


function lp = log_uni_gauss(z, mu, sigma)
    tmp = ( ((z-mu)./sigma).^2 + log(sigma)*2.0 + log(2*pi) )/2.0;
    lp = -tmp;
end


function fv = nonlinear(theta)
    tsq2 = theta.^2;
    fv = (tsq2(1)-1)*tsq2(1)*3.0 + tsq2(2);
end


%theta = ones(2,1)/2;
%y = 1.0;
%sigma = 0.5;
%fv = InformativeLogPosterior(theta, y, sigma)
