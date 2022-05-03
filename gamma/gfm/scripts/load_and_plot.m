function dummy = load_and_plot(in_path,dataset_name, nFactors)
K = nFactors
%algos={'Adadelta without splitting','Adadelta with splitting','Prox', 'Prox2'};
file_name=sprintf('%s/%s_%d_time.mat',in_path, dataset_name, nFactors)
load(file_name);
time_res=time_cache;
file_name=sprintf('%s/%s_%d_loss.mat',in_path, dataset_name, nFactors)
load(file_name);


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


%method_id = [1, 4];
method_id = [1, 3]
method_name = { 'Knowles', 'CVI-MC' };
color_choice=['k','r','g','c','m','b','y'];
marker_choice=['s','d','x','+','*','o','^'];
idx = [1 10 20 50 100];

figure(1);
clf;
plots=[];
shift=1e-12;
for i = 1:length(method_id)
	ii = method_id(i);
	switch ii
	case 1
	%xp=allres.it1;
	yp=-allres.test1;
	case 2
	%xp=allres.it2;
	yp=-allres.test2;
	case 3
	%xp=allres.it3;
	yp=-allres.test3;
	case 4
	%xp=allres.it4;
	yp=-allres.test4;
	otherwise
	error('do not support')
	end
	xp=time_res(:,ii)+shift;
        plot(xp,yp, '-', 'color', color_choice(i), 'linewidth', 3); 
        hold on; 
        pl=plot(xp(idx),yp(idx), 'o', 'color', color_choice(i), 'markersize', markerSize,  'linewidth', 3, 'marker', marker_choice(i), 'markerEdgecolor', color_choice(i), 'markerFaceColor', [1 1 1]);
        plots(i)=pl;

end

grid on;
hold off;
ylim([40,120]);
xlim([0,350]);

ht = title(dataset_name,'interpreter','none');
%hx = xlabel('# pass');
hx = xlabel('Seconds');
hy = ylabel('Test Log Loss');


set(gca, 'fontsize', fontSize3);
set([hx hy],'fontname','avantgarde','fontsize',fontSize1,'color',[.3 .3 .3]);
set(ht,'fontname','avantgarde','fontsize',fontSize1,'color','k','fontweight','bold');

legend([plots], method_name);
name=sprintf('%s_%d.pdf',dataset_name,K);
%print(name, '-dpdf')
set(gcf, 'PaperPosition', [0 0 ww hh]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [ww hh]); %Set the paper to have width 5 and height 5.
saveas(gcf, name, 'pdf') %Save figure
