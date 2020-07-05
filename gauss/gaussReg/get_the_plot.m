clear all; close all;
fontSize1 = 20;
fontSize2 = 22;
fontSize3 = 20;
maxIters = 100000;
B = 1;
lls = zeros(maxIters,2);
dataset_name = 'abalone_scale'

path = '.';

fname = sprintf('%s/blinreg-%s-exact.mat', path, dataset_name)
load(fname);
exact_elbo = exact_obj;

method='iBayesLRule';
option = 'first';
ss = 0.0001
fname = sprintf('%s/blinreg-%s-%s-%s-%.8f.mat', path, dataset_name, method, option, ss)
load(fname);
lls(:,1) = exact_obj + infos(:,1);
options{1} = sprintf('iBayesLRule-rep')

method='cvi';
option = 'first';
ss = 0.00004
fname = sprintf('%s/blinreg-%s-%s-%s-%.8f.mat', path, dataset_name, method, option, ss)
load(fname);
lls(:,2) = exact_obj +infos(:,1);
options{2} = sprintf('BayesLRule-rep')

method='bbvi';
option = 'first';
ss = 0.01
fname = sprintf('%s/blinreg-%s-%s-%s-%.8f.mat', path, dataset_name, method, option, ss)
load(fname);
lls(:,3) = exact_obj +infos(:,1);
options{3} = sprintf('BBVI-rep')

method='vogn';
option = 'first';
ss = 0.0002
fname = sprintf('%s/blinreg-%s-%s-%s-%.8f.mat', path, dataset_name, method, option, ss)
load(fname);
lls(:,4) = exact_obj +infos(:,1);
options{4} = sprintf('VOGN')


total_lls = lls;

title_name = sprintf('abalone');
algos=options;

num_method=length(algos);
color_choice={'r','k','b','g','c','m','y',[0.8 0.8 1]};
marker_choice=['s','d','x','+','*','o','^','.'];
figure(1);
clf;
plots=[];
for ii = 1:num_method
    xp=(1:maxIters)'; yp=total_lls(:,ii);
    assert( size(xp,1) == size(yp,1) )
    idx = yp>0.;

    xp=xp(idx); yp=yp(idx);

    loglog(xp,yp, '-', 'color', color_choice{ii}, 'linewidth', 8);
    hold on;
    pl=plot(xp(1),yp(1), 'o', 'MarkerSize',6,'color', color_choice{ii}, 'linewidth', 8, 'marker', marker_choice(ii), 'markerEdgecolor', color_choice{ii}, 'markerFaceColor', [1 1 1]);
    plots(ii)=pl;
end
grid on
hold off;
hx = xlabel('# Iteration');
hy = ylabel('$ \mathcal{L} - \mathcal{L}^* $', 'Interpreter','latex');
switch dataset_name
case 'abalone_scale'
ylim( [0.01, 100000])
set(gca, 'YTick', [0.1  10^1 10^3 10^5 ])
end
file_name = sprintf('%s',dataset_name);
ht = title(title_name, 'Interpreter', 'none')
hl = legend([plots], algos, 'Location','southwest', 'Interpreter', 'none');
set(hl, 'Color','none');  % =fully transparent

xt = get(gca, 'XTick');
xtkvct = 10.^linspace(1, 10*size(xt,2), 10*size(xt,2));
set(gca, 'XTick', xtkvct);
set(gca, 'XMinorTick','on', 'XMinorGrid','on')

set(gca, 'fontsize', fontSize3);
set([hx hy],'fontname','avantgarde','fontsize',fontSize1,'color',[.3 .3 .3]);
set(ht,'fontname','avantgarde','fontsize',fontSize1,'color','k','fontweight','bold');
name=sprintf('%s.pdf',file_name)

%print(name, '-dpdf', '-fillpage')
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
export_fig(name, '-transparent', '-pdf')
%saveas(gca, name, 'pdf')
