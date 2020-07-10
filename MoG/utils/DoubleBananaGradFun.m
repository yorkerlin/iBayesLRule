function [grad, hess] = DoubleBananaGradFun(theta,y,s1,s2)
theta1sq2 = theta(1)^2;
theta2sq2 = theta(2)^2;

b = (1.0-theta(1))^2 + 100.*(theta(2)-theta1sq2)^2;
F = log(b);
gbTheta1 = (theta(1)-1.0)*2.0 + (theta1sq2-theta(2))*200.0*(2.0*theta(1));
gbTheta2 = 200.*(theta(2)-theta1sq2);

gradLpdTheta1=-( theta(1)/s1 +(F-y)*(gbTheta1/b)/s2 );
gradLpdTheta2=-( theta(2)/s1 +(F-y)*(gbTheta2/b)/s2 );

grad = [gradLpdTheta1; gradLpdTheta2];
if nargout>1
    hess=zeros(2,2);

    gbTheta1sq2 = 2.0 + (theta1sq2-theta(2))*400.0 + 200.0*(4.0*theta1sq2);
    gbTheta2sq2 = 200.0;
    gbTheta1Theta2 = -400.0*theta(1);

    tmp1 = (gbTheta1/b)^2;
    tmp2 = (gbTheta2/b)^2;
    tmp3 = (gbTheta1/b)*(gbTheta2/b);
    hess(1,1) = -(1.0/s1 + tmp1/s2 + (F-y)/s2*(gbTheta1sq2/b -tmp1));
    hess(1,2) = -( tmp3/s2 + (F-y)/s2*( gbTheta1Theta2/b - tmp3 )  );
    hess(2,1) =hess(1,2);
    hess(2,2) = -(1.0/s1 + tmp2/s2 + (F-y)/s2*(gbTheta2sq2/b -tmp2));
end

end
