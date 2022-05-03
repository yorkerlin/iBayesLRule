function dummy = my_gamma_plot(in_path)
K = 39
file_name=sprintf('%s/cytof_%d_loss.mat',in_path,K)
load(file_name)
algos={'RGVI', 'CVI', 'Adam'};
%fontSize1 = 28; 
%fontSize2 = 22; 
%fontSize3 = 22; 
%markerSize = 18; 
fontSize1 = 60; 
fontSize2 = 22; 
fontSize3 = 55; 
markerSize = 40; 

ww = 15;
hh = 12;

num_method=3;
color_choice=['r','k','b','g','c','m','y'];
marker_choice=['s','d','x','+','*','o','^'];
figure(1);
clf;
plots=[];
for ii = 1:num_method
	switch ii
	case 1
	xp=allres.it1; yp=-allres.test1;
	case 2
	xp=allres.it2; yp=-allres.test2;
	case 3
	xp=allres.it3; yp=-allres.test3;
	case 4
	xp=allres.it4; yp=-allres.test4;
	case 5
	xp=allres.it5; yp=-allres.test5;
	otherwise
	error('do not support')
	end
        plot(xp,yp, '-', 'color', color_choice(ii), 'linewidth', 5);
        hold on;
        pl=plot(xp(1),yp(1), 'o', 'color', color_choice(ii), 'linewidth', 5, 'marker', marker_choice(ii), 'markerEdgecolor', color_choice(ii), 'markerFaceColor', [1 1 1]);
        plots(ii)=pl;

end
grid on;
hold off;
ylim([40,120]);

ht = title('Gamma MF','interpreter','none');
hx = xlabel('# pass');
hy = ylabel('Test Log Loss');
legend([plots], algos);

set(gca, 'fontsize', fontSize3);
set([hx hy],'fontname','avantgarde','fontsize',fontSize1,'color',[.3 .3 .3]);
set(ht,'fontname','avantgarde','fontsize',fontSize1,'color','k','fontweight','bold');

name=sprintf('GAMMA_%d.pdf',K);
set(gcf, 'PaperPosition', [0 0 ww hh]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [ww hh]); %Set the paper to have width 5 and height 5.
set(gca, 'Color', 'none'); % Sets axes background
set(gcf, 'Color', 'none'); % Sets axes background
print(name, '-dpdf')
%export_fig(name, '-transparent', '-pdf')
