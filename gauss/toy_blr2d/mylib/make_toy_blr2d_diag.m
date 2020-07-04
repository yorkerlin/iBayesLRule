% Load the toy example experiment data.
addpath(genpath('.'))
clear all;
file_name = strcat('./toy_example_experiment_data.mat');
load(file_name)
[n,n]=size(w1);
t = maxIters;


method = {'iBayesLRule', 'Vadam', 'VOGN'};
colors={'k','b','r','c','g','m','y',[0.8 0.8 1]};
f = figure('Position', [50,50,1000,715]); clf;
% exact post
contourf(w1,w2,reshape(post,[n,n]),5);
cbh=colorbar;
colormap(gray);
hold on
% map estimate
h(1) = plot(wmap(1),wmap(2),'+','color', colors{1}, 'MarkerSize',7, 'linewidth', 8, 'markersize', 7);
h(2) = plot_gaussian_ellipsoid(w_exact_vi, C_exact_vi, 1);
set(h(2), 'color', colors{2}, 'linestyle', '-.', 'linewidth', 12);
% mf-exact
axis([0 20 0 12]);
plot(w_exact_vi(1), w_exact_vi(2), 'o', 'color', colors{2}, 'linewidth', 8, 'markerfacecolor', colors{2}, 'markersize', 5);

for m = 1:length(method)
    plot(w_all(1,t,m), w_all(2,t,m), 'or', 'color', colors{m+2}, 'linewidth', 8, 'markerfacecolor', colors{m+2}, 'markersize', 5);
    h(m+2) = plot_gaussian_ellipsoid(w_all(:,t,m), Sigma_all(:,:,t,m), 1);
    set(h(m+2), 'color', colors{m+2}, 'linewidth', 12);
end


hl = legend(h, {'MAP', 'MF-exact', 'iBayesLRule-diag',  'Vadam', 'VOGN-diag'}, 'location', 'northwest');
hx = xlabel('w_1');
hy = ylabel('w_2');
set(gca, 'fontsize', 24);
%set([hx,hy], 'fontsize', 24, 'fontname', 'helvetica');
set(hl, 'fontsize', 24, 'fontname', 'helvetica');
set(gca, 'xtick', [0:5:20], 'ytick', [0:5:10], 'tickdir', 'out');
set(cbh,'YTick',[0:1e-3:5e-3])

h = gcf;

f.Position(3) = 1000;
f.Position(4) = 800;


axesObjs = get(h, 'Children');  %axes handles
dataObjs = get(axesObjs, 'Children'); %handles to low-level graphics objects in axes

cell_obj = dataObjs(3, 1);
cell_obj = cell_obj{1};

vogn_dot = cell_obj(3);
vogn_line = cell_obj(2);

vadam_dot = cell_obj(5);
vadam_line = cell_obj(4);

vi_dot = cell_obj(1);
vi_line = cell_obj(6);
map_dot = cell_obj(7);

linewidth = 12;

%set(vogn_dot,'color', 'b')
%set(vogn_line,'color', 'b')
set(vogn_line,'linewidth', linewidth)

%set(vadam_dot, 'color', 'r')
%set(vadam_line,'color', 'r')
set(vadam_line,'linewidth', linewidth)

%set(vi_dot,'color', [0, 1.0, 0])
%set(vi_line,'color', [0, 1.0, 0])
set(vi_line,'linewidth', linewidth)

set(map_dot, 'markerSize', 22)
%set(map_dot, 'color', 'black')

%%%%
%fontsize = 30;
%ax = gca;
%set(ax, 'fontsize', fontsize);

ax.XLim = [0, 20];
ax.YLim = [0, 11];
%ax.XLabel.String = 'w1';
%ax.YLabel.Stribeng = 'w2';
%%%
hLegend = findobj(gcf, 'Type', 'Legend');
hLegend.FontSize = 30;
%set(ax, 'LooseInset', get(ax, 'TightInset'));


grid off


set(f,'Units','Inches');
pos = get(f,'Position');
set(f,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])


set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
export_fig('./toy_blr2d.pdf', '-transparent')
%print(f, './toy_blr2d.pdf', '-dpdf', '-fillpage')
