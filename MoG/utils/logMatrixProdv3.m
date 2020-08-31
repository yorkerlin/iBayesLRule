function [res,sig]= logMatrixProdv3(logw,x)
assert( size(logw,2) == size(x,1) )
assert( size(x,2) == 1 )
mask = sign(x);

num_pos=length(x(mask>0));
if num_pos>0
    pos_array = logMatrixProdv3_Helper( logw(:, mask>0), x(mask>0) );
end

num_neg=length(x(mask<0));
if num_neg>0
    neg_array = logMatrixProdv3_Helper( logw(:, mask<0), -x(mask<0) );
end


if num_pos>0 && num_neg>0
    [res, sig] = logsubexpv2(pos_array, neg_array);
elseif num_pos>0
    res = pos_array;
    sig = ones( size(logw,1), size(x,2) );
else
    res = neg_array;
    sig = -ones( size(logw,1), size(x,2) );
end

end


function res = logMatrixProdv3_Helper(logw,x)
assert( size(logw,2) == size(x,1) )
assert( size(x,2) == 1 )

tmp = logw + repmat( log(x), 1, size(logw,1) )';
max_v = max(tmp, [] , 2);
res = log1p(sum(exp(tmp - max_v), 2)-1.0) + max_v;
end


