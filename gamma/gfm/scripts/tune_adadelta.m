

rhorange=1-(10.^-(.5:.25:2));
forgetrange=10.^-[0 .3 (.5:.5:4)]; 
ii=6; 
K=5; 
finalELBO={};
parfor rhoIndex=1:length(rhorange)
    rho=rhorange(rhoIndex);
    finalELBOtemp=[]; 
    for forgetIndex=1:length(forgetrange)
        settings=gammaSGVBsettings(ii);
        settings.samples=1000; 
        settings.rho=rho; 
        settings.forgetting=forgetrange(forgetIndex); 
        [~,ll]=networkModel(X,K,missing,settings); 
        finalELBOtemp(forgetIndex)=ll(end); 
    end
    finalELBO{rhoIndex}=finalELBOtemp; 
end

% optionally save parameters
%save('nipsTestK5.mat','finalELBO'); 

temp=cellfun(@transpose,finalELBO,'UniformOutput',0);

elbos=[ temp{:} ]; 

imagesc(elbos); colorbar(); 
set(gca,'XTickLabel',rhorange);
set(gca,'YTickLabel',forgetrange);
xticks=arrayfun(@(x) sprintf('%2.3f',x),rhorange,'uni',false).'
yticks=arrayfun(@(x) sprintf('%2.3f',x),forgetrange,'uni',false).'

% make heatmap for Figure 2
heatmap(elbos/1e3,xticks,yticks,'%2.2f'); 
xlabel('Adadelta rho'); ylabel('1-momentum'); colorbar(); 
