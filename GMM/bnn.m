close all; clear all;
clc
setSeed(1);

% settings
nrSteps=2e4;

% define log likelihood
y = 1;
sigma = 0.5;
wt = sigma * sigma;

logp0 = @(theta)InformativeLogPosterior(theta,y,sigma);
logPostDens = @(x)InformativeLogPosterior(x,y,sigma);
logPostDensQuad = @(x1,x2)InformativeLogPosteriorQuad(x1,x2,y,sigma);
gradfun = @(x)InformativeGradFun(x,y,sigma,wt);

% contour grid
[theta1,theta2] = meshgrid(-2.0:.005:2,-2.0:0.005:2);
[n,n]=size(theta1);
Z=zeros(size(theta1));

% contour plot true density
disp('Plotting true density')
figure()
subplot(3,3,9)
for i=1:size(theta1,1)
    for j=1:size(theta1,2)
        Z(i,j)=logPostDens([theta1(i,j);theta2(i,j)]);
    end
end
mzr = max(max(Z));
normconst = dblquad(@(x1,x2)exp(logPostDensQuad(x1,x2)-mzr),-2.0,2.0,-2.0,2.0);
Z=Z-mzr-log(normconst);
[C,h] = contour(theta1,theta2,exp(Z)); colormap jet;
title('exact')
xlabel('$z_1$', 'Interpreter','latex');
ylabel('$z_2$', 'Interpreter','latex');
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
true_dist.theta1= theta1;
true_dist.theta2= theta2;
true_dist.Z =Z;



num_com = 8;
normPostDens = @(x1,x2)exp(logPostDensQuad(x1,x2)-mzr-log(normconst));
logNormPostDens = @(x1,x2)(logPostDensQuad(x1,x2)-mzr-log(normconst));


likelihoodFun = @(xSampled,iter)evaluate_likelihood2(xSampled,iter,logPostDens,gradfun);

init_m=zeros(2,1);
init_P=eye(2)*10;

nrSamples = 800000;
k = length(init_m);
rawNorm = randn(k,nrSamples);
rawUnif = rand(nrSamples,1);

results=cell(num_com,3);


% try multiple mixture components
for i=1:num_com
    nrComponents=i;
    trueC = i*10;
    disp(['Estimating approximation with ' int2str(trueC) ' components.'])

    %disp('iBayesLRule second order')
    %method_name = 'iBayesLRule_hess_BNN';
    %option='hess';
    %[mixWeights,mixMeans,mixPrecs]=mix_gauss_iBayesLRule(option,likelihoodFun,trueC,nrSteps,init_m,init_P,floor(nrSteps/50), 30, 0.008, 0.05);


    disp('iBayesLRule first order')
    method_name = 'iBayesLRule_first_order_BNN';
    option='first';
    [mixWeights,mixMeans,mixPrecs] = mix_gauss_iBayesLRule(option,likelihoodFun,trueC,nrSteps,init_m,init_P,floor(nrSteps/50), 50, 0.006, 0);

    % approximation density
    myDens=@(x1,x2)DensApproximation(x1,x2,mixWeights,mixMeans,mixPrecs);
    myLogDens=@(x1,x2)log(myDens(x1,x2));

    %LDens=@(x)Get_Conditional_Log_Prbs(x,log(mixWeights),mixMeans,mixPrecs);

    % contour plot approximation
    subplot(3,3,i)
    for l=1:size(theta1,1)
        for j=1:size(theta1,2)
            Z(l,j)=myLogDens(theta1(l,j),theta2(l,j));
        end
    end

    [C,h] = contour(theta1,theta2,exp(Z));
    colormap jet
    xlabel('$z_1$', 'Interpreter','latex');
    ylabel('$z_2$', 'Interpreter','latex');
    title(int2str(trueC))
    set(gca, 'Color', 'none'); % Sets axes background
    set(gcf, 'Color', 'none'); % Sets axes background
    results{nrComponents,1} = theta1;
    results{nrComponents,2} = theta2;
    results{nrComponents,3} = Z;
end

file_name = sprintf('bnn.mat')
save(file_name, 'results','true_dist');

ww = 15;
hh = 12;
set(gcf, 'PaperPosition', [0 0 ww hh]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [ww hh]); %Set the paper to have width 5 and height 5.
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
export_fig(method_name, '-transparent', '-pdf')
%print(file_name, '-dpdf');


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

