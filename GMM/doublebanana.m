close all; clear all;
clc
setSeed(1);

% settings
nrSteps=6e4;

% define log likelihood
y = log(30);
s1 = 1.0;
s2 = 0.09;
nd1 = 5;
nd2 = 3;

logp0 = @(theta)DoubleBananaLogPosterior(theta,y,s1,s2);
logPostDens = @(x)DoubleBananaLogPosterior(x,y,s1,s2);
logPostDensQuad = @(x1,x2)DoubleBananaLogPosteriorQuad(x1,x2,y,s1,s2);
gradfun = @(x)DoubleBananaGradFun(x,y,s1,s2);

% contour grid
[theta1,theta2] = meshgrid(-2.0:.005:2,-1.0:0.002:2);
[n,n]=size(theta1);
Z=zeros(size(theta1));

% contour plot true density
disp('Plotting true density')
figure()
subplot(nd1,nd2,nd1*nd2)
for i=1:size(theta1,1)
    for j=1:size(theta1,2)
        Z(i,j)=logPostDens([theta1(i,j);theta2(i,j)]);
        %[g,H] = gradfun([theta1(i,j);theta2(i,j)]);
    end
end
mzr = max(max(Z));
normconst = dblquad(@(x1,x2)exp(logPostDensQuad(x1,x2)-mzr),-2.0,2.0,-1.0,2.0);
Z=Z-mzr-log(normconst);
[C,h] = contour(theta1,theta2,exp(Z)); colormap jet;
title('exact')
%xlabel('x');
%ylabel('y');
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
true_dist.theta1= theta1;
true_dist.theta2= theta2;
true_dist.Z =Z;

num_com = nd1*nd2-1;
normPostDens = @(x1,x2)exp(logPostDensQuad(x1,x2)-mzr-log(normconst));
logNormPostDens = @(x1,x2)(logPostDensQuad(x1,x2)-mzr-log(normconst));

callback= @(iter,post_dist)dummy(iter,post_dist);

likelihoodFun = @(xSampled,iter)evaluate_likelihood2(xSampled,iter,logPostDens,gradfun);

init_m=zeros(2,1);
init_P=eye(2);

nrSamples = 600000;
k = length(init_m);
rawNorm = randn(k,nrSamples);
rawUnif = rand(nrSamples,1);

results=cell(num_com,3);


% try multiple mixture components
for i=1:num_com
    setSeed(1);
    nrComponents=i;
    %disp(['Estimating approximation with ' int2str(i) ' components.'])
    %option='hess'
    %[q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_cvi_new(option,likelihoodFun,nrComponents,50,init_m,init_P,100,100,0.0005, 0.01);

    %method_name = 'cvi_hess_doublebanana2d';
    %option='hess'
    %[mixWeights,mixMeans,mixPrecs] = mix_gauss_cvi_new(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P,floor(nrSteps/50), 100, 0.004, 0.01, 'doublebanana',callback, init_post);

    %disp('iBayesLRule second order')
    %method_name = 'iBayesLRule_hess_doublebanana2d';
    %option='hess'
    %[mixWeights,mixMeans,mixPrecs] = mix_gauss_iBayesLRule(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P,floor(nrSteps/50), 100, 0.003, 0.01);


    disp('iBayesLRule first order')
    method_name = 'iBayesLRule_first_order_doublebanana2d';
    option='first'
    [mixWeights,mixMeans,mixPrecs] =mix_gauss_iBayesLRule(option,likelihoodFun,nrComponents,nrSteps,init_m,init_P,floor(nrSteps/50), 50, 0.0015, 0.01);


    % approximation density
    myDens=@(x1,x2)DensApproximation(x1,x2,mixWeights,mixMeans,mixPrecs);
    myLogDens=@(x1,x2)log(myDens(x1,x2));

    %LDens=@(x)Get_Conditional_Log_Prbs(x,log(mixWeights),mixMeans,mixPrecs);

    % contour plot approximation
    subplot(nd1,nd2,i)
    for l=1:size(theta1,1)
        for j=1:size(theta1,2)
            Z(l,j)=myLogDens(theta1(l,j),theta2(l,j));
        end
    end


    [C,h] = contour(theta1,theta2,exp(Z));
    colormap jet
    if i==14
    xlabel('x');
    end
    if i==7
    ylabel('y');
    end
    title(int2str(i))
    set(gca, 'Color', 'none'); % Sets axes background
    set(gcf, 'Color', 'none'); % Sets axes background
    results{nrComponents,1} = theta1;
    results{nrComponents,2} = theta2;
    results{nrComponents,3} = Z;
end

file_name = sprintf('doublebanana2d.mat')
save(file_name, 'results','true_dist');

ww = 15;
hh = 12;
set(gcf, 'PaperPosition', [0 0 ww hh]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [ww hh]); %Set the paper to have width 5 and height 5.
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
%print(file_name, '-dpdf');
export_fig(method_name, '-transparent', '-pdf')


function res = dummy(iter,post_dist)
res = 0;
end


function [lp,grad,hess] = evaluate_likelihood2(xSampled,iter,logPostFun,gradfun)
%iter is a dummy index in this example
[k,nrSamples] = size(xSampled);
lp = zeros(nrSamples,1);
grad = zeros(k,nrSamples);
if nargout>2
    hess = zeros(k,k,nrSamples);
end

for j=1:nrSamples
    lp(j) = logPostFun(xSampled(:,j));
    if nargout>2
        [grad(:,j),hess(:,:,j)] = gradfun(xSampled(:,j));
    else
        grad(:,j) = gradfun(xSampled(:,j));
    end
end
end


