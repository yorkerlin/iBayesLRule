function wx = weightProd2(logW,x)
    if length(logW)>1
        assert(  size(logW,1) == size(x, length(size(x)) )  )
        logW = reshape(logW, [ones(1,length(size(x))-1),size(logW,1)]);
    end
    tmp = x;
    sigInd = sign(tmp);
    wx=sigInd.*exp(logW+log(abs(tmp)));
    wx(sigInd==0)=0;
end

