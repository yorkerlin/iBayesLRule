nrSteps = 2e6;
setSeed(0);
rng(0, 'twister');
p_alpha = 1.0;%degree of freedom (2*p_alpha)

d=20; %dim=20

%method = 'iBayesLRule-first'; %the re-parametrization trick
%stepSize = 0.004;

method = 'iBayesLRule-hess'; %the hessian trick
stepSize = 0.04;

decay_mix = 0;

init_P = eye(d)/300; %initial precision should be small to avoid stuck in one mode
init_m = zeros(d,1);

p_nrComponents = 10;
p_mixMeans = cell(p_nrComponents,1);
p_mixPrecs = cell(p_nrComponents,1);
p_logMixWeights = ones(1,p_nrComponents)*(-log(p_nrComponents));

offset = 20;
max_mean = -2*offset;
min_mean = 2*offset;

for c=1:p_nrComponents
    p_mixMeans{c} = rand(d,1)*(2*offset)-offset;
    %p_mixPrecs{c} = eye(d);
    A = randn(d,d)*(0.1*d);
    p_mixPrecs{c} = A'*A+eye(d);

    max_mean = max(p_mixMeans{c}, max_mean);
    min_mean = min(p_mixMeans{c}, min_mean);
end
true_dist.mixWeights = exp(p_logMixWeights);
true_dist.mixMeans = p_mixMeans;
true_dist.mixPrecs = p_mixPrecs;
true_dist.alpha = p_alpha;

global track_restuls;
track_restuls = cell(nrSteps,1);
global track_index;
track_index = zeros(nrSteps,1);

if d==2
    Range = 10;
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
    loglikelihoodT =@(xSampled, Mean, Prec)log_T(xSampled, Mean, Prec, true_dist.alpha);
    [~,~,logMarPrb] = Get_Conditional_Log_Prbsv2(W_tr,p_logMixWeights,p_mixMeans,p_mixPrecs,loglikelihoodT);

    post = exp(logMarPrb)';
    callback= @(iter,post_dist)plotFun2(method,iter,true_dist,post_dist,w1,w2,W_tr,n1,n2,post,axis_x,axis_y);
else
    callback= @(iter,post_dist)plotFun(method,iter,true_dist,post_dist);
end


likelihood =@(xSampled,iter)Log_Mix_T(xSampled,p_logMixWeights,p_mixMeans,p_mixPrecs,true_dist.alpha);

q_nrComponents=25;
preIt=floor(nrSteps/1000);
dataset_name = sprintf('toyT-%d',d);

nrSamples = 10; %MC samples

option = 'hess';
q_mixMeans = cell(q_nrComponents,1);
q_mixPrecs = cell(q_nrComponents,1);

for c=1:q_nrComponents
    q_mixMeans{c} = randn(d,1)*100;
    q_mixPrecs{c} = init_P;
end
init_post.mixMeans = q_mixMeans;
init_post.mixPrecs = q_mixPrecs;

%do some warm-up for all methods
dummycallback=@(iter,post_dist)dummyFun(iter,post_dist);
[q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_cvi_new(option,likelihood,q_nrComponents,100,init_m,init_P,preIt,100,0.0000001,0.01,dataset_name,dummycallback,init_post);

init_post.mixMeans = q_mixMeans;
init_post.mixPrecs = q_mixPrecs;
init_post.mixWeights = ones(1,q_nrComponents)/(q_nrComponents);
save('init_GMM20d', 'init_post');
disp('running')

file_name = sprintf('GMM-%d_d-trace-%s-%d-%.8f-%.8f.mat',d,method,q_nrComponents,stepSize,decay_mix)


switch method
case {'cvi-hess'}
    option = 'hess';
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_cvi_new(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback, init_post);
case {'cvi-first'}
    option = 'first';
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_cvi_new(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback, init_post);
case {'bbvi-hess'}
    option = 'hess';
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_bbvi(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback, init_post);
case {'bbvi-first'}
    option = 'first';
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_bbvi(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback, init_post);
case {'iBayesLRule-hess'}
    option = 'hess';
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_iBayesLRule(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback, init_post);
case {'iBayesLRule-first'}
    option = 'first';
    [q_mixWeights,q_mixMeans,q_mixPrecs] = mix_gauss_iBayesLRule(option,likelihood,q_nrComponents,nrSteps,init_m,init_P,preIt, nrSamples, stepSize, decay_mix, dataset_name,callback, init_post);
otherwise
    error('no such method')
end


save(file_name,'track_restuls','track_index', 'true_dist');

function dummy=dummyFun(iter,post_dist)
    dummy = 0;
end

function dummy=plotFun(method, iter,true_dist,post_dist)
    global track_restuls;
    global track_index;
    nrComponents = size(post_dist.mixMeans,1);
    d =length(post_dist.mixMeans{1});
    track_restuls{iter} = post_dist;
    track_index(iter) =iter;

    if iter>1000
    [true_uni_mean_info, true_uni_prec_info, true_uni_max_x, true_uni_min_x] = get_uni_infos(true_dist.mixWeights,true_dist.mixMeans,true_dist.mixPrecs,10);

    [uni_mean_info, uni_prec_info, uni_max_x, uni_min_x] = get_uni_infos(post_dist.mixWeights,post_dist.mixMeans,post_dist.mixPrecs);

    figure(1);
    loglikelihoodT =@(xSampled, Mean, Prec)log_T(xSampled, Mean, Prec, true_dist.alpha);
    for k=1:d
        xSamples = linspace(true_uni_min_x{k}, true_uni_max_x{k}, 10000);
        [~, ~, uni_logMarPrb] = Get_Conditional_Log_Prbs(xSamples, log(post_dist.mixWeights), uni_mean_info{k}, uni_prec_info{k});
        figure(k);
        plot(xSamples, exp(uni_logMarPrb), 'color', 'r');
        hold on;
        [~,~,true_uni_logMarPrb] = Get_Conditional_Log_Prbsv2(xSamples, log(true_dist.mixWeights), true_uni_mean_info{k}, true_uni_prec_info{k},loglikelihoodT);
        plot(xSamples, exp(true_uni_logMarPrb), 'color', 'b');
        legend({'approx', 'true'})
        title(sprintf('%s Dim %d at iter %d',method,k,iter))
        hold off;
    end

    drawnow
    end
    dummy = 0;
end


function dummy=plotFun2(method, iter,true_dist,post_dist,w1,w2,W_tr,n1,n2,post,axis_x,axis_y)
    figure(1);
    nrComponents = size(post_dist.mixMeans,1);

    % exact post
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
    figure(2);
    [~,~,logMarPrb] = Get_Conditional_Log_Prbs(W_tr,log(post_dist.mixWeights),post_dist.mixMeans,post_dist.mixPrecs);
    post = exp(logMarPrb)';
    Z = reshape(post,[n1,n2]);


    contour(w1,w2,Z,10); colorbar;
    hold on
    grid on
    axis([axis_x(1) axis_x(2) axis_y(1) axis_y(2)]);
    hold off;

    d =length(post_dist.mixMeans{1});

    [true_uni_mean_info, true_uni_prec_info, true_uni_max_x, true_uni_min_x] = get_uni_infos(true_dist.mixWeights,true_dist.mixMeans,true_dist.mixPrecs,10);

    [uni_mean_info, uni_prec_info, uni_max_x, uni_min_x] = get_uni_infos(post_dist.mixWeights,post_dist.mixMeans,post_dist.mixPrecs);

    loglikelihoodT =@(xSampled, Mean, Prec)log_T(xSampled, Mean, Prec, true_dist.alpha);
    for k=1:d
        xSamples = linspace(true_uni_min_x{k}, true_uni_max_x{k}, 10000);
        [~, ~, uni_logMarPrb] = Get_Conditional_Log_Prbs(xSamples, log(post_dist.mixWeights), uni_mean_info{k}, uni_prec_info{k});
        figure(2+k);
        plot(xSamples, exp(uni_logMarPrb), 'color', 'r');
        hold on;
        [~,~,true_uni_logMarPrb] = Get_Conditional_Log_Prbsv2(xSamples, log(true_dist.mixWeights), true_uni_mean_info{k}, true_uni_prec_info{k},loglikelihoodT);
        plot(xSamples, exp(true_uni_logMarPrb), 'color', 'b');
        legend({'approx', 'true'})
        title(sprintf('%s Dim %d',method,k))
        hold off;
    end

    drawnow
    dummy = 0;
end

