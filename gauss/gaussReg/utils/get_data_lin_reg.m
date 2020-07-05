function [y, X, y_te, X_te] = get_data_lin_reg(name, seed)
switch name
case {'housing_scale','mpg_scale','abalone_scale','mg_scale','cpusmall_scale', 'bodyfat_scale','cadata'}
  setSeed(seed);
  load(name);
  [N,D] = size(X);
  X = [ones(N,1) X];
  [X, y, X_te, y_te] = split_data(y, X, 0.8);
otherwise
    error('unknown dataset')
end
end
function [XTr, yTr, XTe, yTe] = split_data(y, X, prop)
N = size(y,1);
idx = randperm(N);
Ntr = floor(prop * N);
idxTr = idx(1:Ntr);
idxTe = idx(Ntr+1:end);
XTr = X(idxTr,:);
yTr = y(idxTr);
XTe = X(idxTe,:);
yTe = y(idxTe);
end
