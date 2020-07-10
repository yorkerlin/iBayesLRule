function lpd = DoubleBananaLogPosterior(theta,y,s1,s2)
theta1sq2 = theta(1)^2;
theta2sq2 = theta(2)^2;
F = log( (1.0-theta(1))^2 + 100.*(theta(2)-theta1sq2)^2 );
lpd = -( (theta1sq2+theta2sq2)/s1 +  (y-F)^2/ s2 )/2.;
