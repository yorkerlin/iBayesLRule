% Run EPM model with NIPS data with six different ways of tuning learning
% rate
likeCurves={}; 
for ii=1:6
    rng(1);
    [X masks]=load_nips(0); 
    K=10; 
    missing=sparse(false(size(X))); 
    settings=gammaSGVBsettings(ii);
    [~,likeCurves{ii}]=networkModel(X,K,missing,settings); 
end

% plot figure 1
close all
clf
subplot(1,2,1); 
hold off
names={'SGD','SGD+momentum','RMSprop','Adagrad','Adadelta','Adadelta+mom'}; 
for ii=1:6
    ln=length(likeCurves{ii}); 
    semilogx( [1 20*(1:ln-2)], -likeCurves{ii}(2:end), 'o-','DisplayName', names{ii}); 
    hold on
end
xlim([0,1000]);
xlabel('iterations'); 
ylabel('negative ELBO');

% ylim([-4e5 1e5])
legend('show','Location','NorthEast');

finalLL=cellfun(@(g) -g(end), likeCurves);
subplot(1,2,2); 
cla
bar( finalLL ); 
set(gca,'XTickLabel',names); 

xticklabel_rotate([],45,[]); %,'Fontsize',14)
ylabel('negative ELBO');

% annoying code to give enough space to the x-axis labels
pos=get(gca,'Position'); 
pos(4)=pos(4)-.1; 
pos(3)=pos(3)+.1
pos(2)=pos(2)+.1;
set(gca,'Position',pos);