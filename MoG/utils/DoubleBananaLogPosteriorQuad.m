function lpd = DoubleBananaLogPosteriorQuad(theta1,theta2,y,s1,s2)
lpd=zeros(size(theta1));
for i=1:length(theta1)
    theta1sq2 = theta1(i)^2;
    theta2sq2 = theta2^2;
    F = log( (1.0-theta1(i))^2 + 100.*(theta2-theta1sq2)^2 );
    lpd(i) = -( (theta1sq2+theta2sq2)/s1 +  (y-F)^2/ s2 )/2.;
end
