function blinreg_save(dataset_name, method)
maxIters = 100000;
B = 1;
option ='first'


switch method
case {'vogn'}
    switch dataset_name
    case {'abalone_scale'}
        sslist = [0.0002]
    otherwise
        error('do not suport')
    end
case {'bbvi'}
    switch dataset_name
    case {'abalone_scale'}
        sslist = [0.01]%abalone
    otherwise
            error('error');
    end
case {'cvi'}
    switch dataset_name
    case {'abalone_scale'}
        sslist = [0.00004]%abalone
    otherwise
            error('error');
    end
case {'iBayesLRule'}
    switch dataset_name
    case {'abalone_scale'}
        sslist = [0.0001]%abalone
    otherwise
            error('error');
    end
end

for ss = sslist
    fname = sprintf('./blinreg-%s-%s-%s-%.8f.mat', dataset_name, method, option, ss)
    [infos] = blinreg(dataset_name,option,method,maxIters,ss,B);
    save(fname,'infos','option','method','dataset_name', 'B', 'ss');
end

end
