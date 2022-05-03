function lp=log1pe(in)
    lp=zeros(size(in)); 
    isLinear=in>10; 
    lp(isLinear)=in(isLinear); 
    lp(~isLinear)=log1p(exp(in(~isLinear)));
end
