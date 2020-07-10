close all; clear all;
clc
format long
setSeed(1);
% settings
nrSteps=2e5;%ok

% define log likelihood
mu1 = 0;
b = 1;

logPostDens = @(x)Laplace2dLogPosterior(x,mu1,b);
logPostDensQuad = @(x1,x2)Laplace2dLogPosteriorQuad(x1,x2,mu1,b);
%gradfun = @(x)LaplaceGradFun_mat(x, mu1, b);

% contour grid
[theta1,theta2] = meshgrid(-3.0:.002:3.0,-3.0:0.002:3.0);
[n,n]=size(theta1);
Z=zeros(size(theta1));

% contour plot true density
disp('Plotting true density')
figure()
subplot(3,2,6)
for i=1:size(theta1,1)
    for j=1:size(theta1,2)
        Z(i,j)=logPostDens([theta1(i,j);theta2(i,j)]);
        %[g,H] = gradfun([theta1(i,j);theta2(i,j)]);
    end
end
mzr = max(max(Z));
normconst = dblquad(@(x1,x2)exp(logPostDensQuad(x1,x2)-mzr),-3.0,3.0,-3.0,3.0);
Z=Z-mzr-log(normconst);
[C,h] = contour(theta1,theta2,exp(Z)); colormap jet;
title('exact','FontSize', 15)
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
true_dist.theta1= theta1;
true_dist.theta2= theta2;
true_dist.Z =Z;

num_com = 5;

likelihoodFun = @(xSampled,iter)evaluate_likelihood2(xSampled,iter, mu1, b);

init_m=zeros(2,1);
init_P=eye(2);

results=cell(num_com,3);


% try multiple mixture components
for i=1:num_com
    disp(['Estimating approximation with ' int2str(i) ' components.'])
    nrComponents=i;

    %method_name = 'cvi_hess_laplace';
    %option='hess'
    %[mixWeights,mixMeans,mixPrecs] = mix_gauss_cvi_new(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P,floor(nrSteps/50), i*2, 0.005, 0.01);

    %disp('iBayesLRule second order')
    %method_name = 'iBayesLRule_hess_laplace';
    %option='hess';
    %[mixWeights,mixMeans,mixPrecs] = mix_gauss_iBayesLRule(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P,floor(nrSteps/50), i*2, 0.005, 0.01);


    disp('iBayesLRule first order')
    method_name = 'iBayesLRule_first_order_laplace';
    option='first';
    [mixWeights,mixMeans,mixPrecs] = mix_gauss_iBayesLRule(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P,floor(nrSteps/50), 25, 0.003, 0.01);%ok

    myDens=@(x1,x2)DensApproximation(x1,x2,mixWeights,mixMeans,mixPrecs);
    myLogDens=@(x1,x2)log(myDens(x1,x2));

    % contour plot approximation
    subplot(3,2,i)
    for l=1:size(theta1,1)
        for j=1:size(theta1,2)
            Z(l,j)=myLogDens(theta1(l,j),theta2(l,j));
        end
    end

    [C,h] = contour(theta1,theta2,exp(Z));
    colormap jet
    if i==5
    xlabel('x');
    end
    if i==3
    ylabel('y');
    end
    title(int2str(i),'FontSize', 15)
    set(gca, 'Color', 'none'); % Sets axes background
    set(gcf, 'Color', 'none'); % Sets axes background
    results{nrComponents,1} = theta1;
    results{nrComponents,2} = theta2;
    results{nrComponents,3} = Z;
end


file_name = sprintf('laplace2d.mat')
save(file_name, 'results','true_dist');

ww = 15;
hh = 12;
set(gcf, 'PaperPosition', [0 0 ww hh]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [ww hh]); %Set the paper to have width 5 and height 5.
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
export_fig(method_name, '-transparent', '-pdf')
%print(file_name, '-dpdf');


function [lp,grad,hess] = evaluate_likelihood2(xSampled,iter, mu1, b)
%iter is a dummy index in this example
[k,nrSamples] = size(xSampled);
lp = zeros(nrSamples,1);
grad = zeros(k,nrSamples);
if nargout>2
    hess = zeros(k,k,nrSamples);
end

lp = LaplaceLogPosterior_mat(xSampled,mu1,b);
grad = LaplaceGradFun_mat(xSampled,mu1,b);
end




function lp = Laplace2dLogPosterior(theta, mu1, b)
assert(length(theta)==2)
res = theta - circshift(theta,1);
lp= -(sum(abs(res(2:end)))+abs(res(1)+theta(end)-mu1) )/b-log(2.0*b)*length(theta);
end

function lpd = Laplace2dLogPosteriorQuad(theta1,theta2,mu1, b)
lpd=zeros(size(theta1));
for i=1:length(theta1)
    theta=[theta1(i), theta2];
    lpd(i) = Laplace2dLogPosterior(theta, mu1, b);
end
end


function grad = LaplaceGradFun_mat(theta, mu1, b)
theta_shift = circshift(theta,1);
theta_shift(1,:) = theta_shift(1,:) - theta(end,:) + mu1;
tmp =( 2*(theta>theta_shift) - 1 );
tmp(theta_shift == theta) = 0;
tmp2=-circshift(tmp,-1);
tmp2(end,:) = 0;
grad = -(tmp + tmp2)/b;
end


function lp = LaplaceLogPosterior_mat(theta, mu1, b)
res = theta - circshift(theta,1);
lp= -(sum(abs(res(2:end,:)),1)+abs(res(1,:)+theta(end,:)-mu1) )/b-log(2.0*b)*size(theta,1);
lp = lp';
end
