close all;
clear all;
setSeed(0);
rng(0, 'twister');

%method='cvi_hess'
%method='iBayesLRule_hess'
method='iBayesLRule_first_order'
%method='bbvi_first_order'
d=2;

init_P = eye(d); %initial precision should be small to avoid stuck in one mode
init_m = zeros(d,1);

p_nrComponents = 5;
p_mixMeans = cell(p_nrComponents,1);
p_mixPrecs = cell(p_nrComponents,1);
p_logMixWeights = ones(1,p_nrComponents)*(-log(p_nrComponents));

offset = 10;
max_mean = -2*offset;
min_mean = 2*offset;

assert(d==2)
theta = 2*pi/p_nrComponents;
U =[ cos(theta), sin(theta);
-sin(theta), cos(theta) ];
for c=1:p_nrComponents
    if c==1
        p_mixMeans{c} =[1.5;0.0];
        p_mixPrecs{c} = diag([1,100]);
    else
        p_mixMeans{c} =U*p_mixMeans{c-1};
        p_mixPrecs{c} =U*p_mixPrecs{c-1}*U';
    end
    max_mean = max(p_mixMeans{c}, max_mean);
    min_mean = min(p_mixMeans{c}, min_mean);
end

true_dist.mixWeights = exp(p_logMixWeights);
true_dist.mixMeans = p_mixMeans;
true_dist.mixPrecs = p_mixPrecs;

if d==2
    Range = 4;
    Step=0.05;
    max_x = max_mean(1) + Range;
    min_x = min_mean(1) - Range;
    max_y = max_mean(2) + Range;
    min_y = min_mean(2) - Range;
    axis_x= [min_x, max_x];
    axis_y= [min_y, max_y];
    [w1,w2]=meshgrid(min_x:Step:max_x,min_y:Step:max_y);
    [n1,n2]=size(w1);
    W=[reshape(w1,n1*n2,1) reshape(w2,n1*n2,1)];
    W_tr = W';
    [~,~,logMarPrb] = Get_Conditional_Log_Prbs(W_tr,p_logMixWeights,p_mixMeans,p_mixPrecs);
    post = exp(logMarPrb)';
    callback= @(iter,post_dist)plotFun2(iter,true_dist,post_dist,w1,w2,W_tr,n1,n2,post,axis_x,axis_y);
else
    callback= @(iter,post_dist)plotFun(iter,true_dist,post_dist);
end

likelihood =@(xSampled,iter)Log_Mix_Gauss(xSampled,p_logMixWeights,p_mixMeans,p_mixPrecs);

q_nrComponents=10;
nrSteps = 1e4;
preIt=floor(nrSteps/50); %plot every preIt iterations
stepSize = 0.01;%bbvi
decay_mix = 0.05;%update for mixing weight
nrSamples = 50; %MC samples
dataset_name = sprintf('toy-%d',d);

disp('running')


switch method
case {'cvi_hess'}
    option='hess'
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_cvi_new(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback);
case {'bbvi_hess'}
    option='hess'
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_bbvi(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback);

case {'bbvi_first_order'}
    option='first'
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_bbvi(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback);

case {'iBayesLRule_hess'}
    option='hess'
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_iBayesLRule(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback);
case {'iBayesLRule_first_order'}
    option='first'
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_iBayesLRule(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback);
otherwise
    error('no such method')
end

method_name = sprintf('%s_star_GMM%d',method, q_nrComponents);
ww = 15;
hh = 12;
set(gcf, 'PaperPosition', [0 0 ww hh]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [ww hh]); %Set the paper to have width 5 and height 5.
export_fig(method_name, '-pdf')
%saveas(gca, method_name, 'pdf')



function dummy=plotFun(iter,true_dist,post_dist)
    figure(1);
    nrComponents = size(post_dist.mixMeans,1);
    d =length(post_dist.mixMeans{1});

    [true_uni_mean_info, true_uni_prec_info, true_uni_max_x, true_uni_min_x] = get_uni_infos(true_dist.mixWeights,true_dist.mixMeans,true_dist.mixPrecs);

    [uni_mean_info, uni_prec_info, uni_max_x, uni_min_x] = get_uni_infos(post_dist.mixWeights,post_dist.mixMeans,post_dist.mixPrecs);

    for k=1:d
        xSamples = linspace(true_uni_min_x{k}, true_uni_max_x{k}, 10000);
        [~, ~, uni_logMarPrb] = Get_Conditional_Log_Prbs(xSamples, log(post_dist.mixWeights), uni_mean_info{k}, uni_prec_info{k});
        figure(k);
        plot(xSamples, exp(uni_logMarPrb), 'color', 'r');
        hold on;
        [~, ~, true_uni_logMarPrb] = Get_Conditional_Log_Prbs(xSamples, log(true_dist.mixWeights), true_uni_mean_info{k}, true_uni_prec_info{k});
        plot(xSamples, exp(true_uni_logMarPrb), 'color', 'b');
        legend({'approx', 'true'})
        title(sprintf('Dim %d',k))
        hold off;
    end

    drawnow
    dummy = 0;
end




function dummy=plotFun2(iter,true_dist,post_dist,w1,w2,W_tr,n1,n2,post,axis_x,axis_y)
    figure(1);
    nrComponents = size(post_dist.mixMeans,1);

    subplot(3,2,1)
    % exact post
    contourf(w1,w2,reshape(post,[n1,n2]),10); colorbar;
    title('exact')

    %figure(2);
    subplot(3,2,2)
    [~,~,logMarPrb] = Get_Conditional_Log_Prbs(W_tr,log(post_dist.mixWeights),post_dist.mixMeans,post_dist.mixPrecs);
    post = exp(logMarPrb)';
    Z = reshape(post,[n1,n2]);


    contour(w1,w2,Z,10); colorbar;
    hold on
    grid on
    axis([axis_x(1) axis_x(2) axis_y(1) axis_y(2)]);
    hold off;
    title('approx')

    subplot(3,2,3)
    contourf(w1,w2,reshape(post,[n1,n2]),10); colorbar;
    lwt = cell(nrComponents+1,1);
    lwt{1} = sprintf('iter-%d',iter);

    ii = (1:nrComponents)';
    colors=num2cell(jet(length(ii)));
    for c=1:nrComponents
        hold on
        h2 = plot_gaussian_ellipsoid(post_dist.mixMeans{c}, inv( post_dist.mixPrecs{c}));
        set(h2, 'color', [colors{nrComponents-c+1,:}], 'linewidth', 3);
        lwt{c+1} = sprintf('%.5f',post_dist.mixWeights(c));
    end
    %legend(lwt)
    grid on
    axis([axis_x(1) axis_x(2) axis_y(1) axis_y(2)]);
    hold off;
    tl = sprintf('fitted using %d components', nrComponents);
    title(tl)

    subplot(3,2,4)
    pie(post_dist.mixWeights)
    title('mixing weights')

    d =length(post_dist.mixMeans{1});

    [true_uni_mean_info, true_uni_prec_info, true_uni_max_x, true_uni_min_x] = get_uni_infos(true_dist.mixWeights,true_dist.mixMeans,true_dist.mixPrecs);

    [uni_mean_info, uni_prec_info, uni_max_x, uni_min_x] = get_uni_infos(post_dist.mixWeights,post_dist.mixMeans,post_dist.mixPrecs);

    for k=1:d
        xSamples = linspace(true_uni_min_x{k}, true_uni_max_x{k}, 10000);
        [~, ~, uni_logMarPrb] = Get_Conditional_Log_Prbs(xSamples, log(post_dist.mixWeights), uni_mean_info{k}, uni_prec_info{k});
        %figure(2+k);
        subplot(3,2,4+k)
        plot(xSamples, exp(uni_logMarPrb), 'color', 'r');
        hold on;
        [~, ~, true_uni_logMarPrb] = Get_Conditional_Log_Prbs(xSamples, log(true_dist.mixWeights), true_uni_mean_info{k}, true_uni_prec_info{k});
        plot(xSamples, exp(true_uni_logMarPrb), 'color', 'b');
        legend({'approx', 'true'})
        title(sprintf('Dim %d',k))
        hold off;
    end

    drawnow
    dummy = 0;
end

