function lpd = InformativeLogPosteriorQuad(theta1,theta2,y, sigma)
lpd=zeros(size(theta1));
for i=1:length(theta1)
    theta = [theta1(i); theta2];
    lpd(i) = InformativeLogPosterior(theta, y, sigma);
end
