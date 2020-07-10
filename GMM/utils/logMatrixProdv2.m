function [res,sig]= logMatrixProdv2(logw,x)
assert( size(logw,2) == size(x,1) )
mask = sign(x);

pos_array = x(mask>0);
num_post=length(pos_array(:));
neg_array = x(mask<0);
num_neg=length(neg_array(:));

%if length(x(:))==num_neg+num_post
    [res,sig]= logMatrixProdv2_Helper(logw,x,mask,num_post,num_neg);
%else
    %{
    %do not work well due to numerical issues if zeros appear in x
    %for example:
    %mat =
    %1000   -7.7787   0
    %-1000  2.0154    0.0797
    %0       -25.5182  -2.1802
    %-1e-8   -8.5385   -4.6652
    %1e-8    -13.2359  -18.3368
    %lw = ones(4,5)
    %logMatrixProdv2(lw,mat) %do not give the right answer

    nonzero=x(mask~=0);
    corr = min(abs(x)+min(abs(nonzero(:)))/2, [], 1);
    corr = repmat(corr, size(x,1),1);

    logw_max = max(logw,[],2);
    cor = logMatrixProdv2(logw, corr);

    [res_tmp, sig_tmp] = logMatrixProdv2(logw,x+corr);
    res = zeros( size(logw,1), size(x,2) );
    sig = zeros( size(logw,1), size(x,2) );

    idx_pos = sig_tmp>0;
    cor_pos = cor(idx_pos);
    res_tmp_pos = res_tmp(idx_pos);
    abs(res_tmp_pos - cor_pos)
    idx = abs(res_tmp_pos - cor_pos)<1e-6; %the difference may be big (this is the issue)
    cor_pos(idx) = res_tmp_pos(idx);

    [r,s] = logsubexp( res_tmp_pos, cor_pos );
    res(idx_pos)=r; sig(idx_pos)=s;

    idx_neg = sig_tmp<0;
    r = logsumexpbinary(res_tmp(idx_neg), cor(idx_neg) );
    res(idx_neg)=r; sig(idx_neg)=-1;

    res(sig_tmp ==0) = cor(sig_tmp==0);
    sig(sig_tmp ==0) = -1;
    %}
%end

end

%function [res] = logsumexpbinary(a,b)
%disp('l')
%ta=a(:)'; tb=b(:)';
%tmp=[ta;tb];
%tv=max(tmp,[],1);
%res= reshape(tv+log1p( sum( exp(tmp-tv)  ) - 1. ), size(a));
%end

function [res,sig]= logMatrixProdv2_Helper(logw,x, mask, num_post, num_neg)
assert( length(x(:)) == num_neg+num_post); %do not deal with zero entries
tmp = log( permute( repmat(abs(x), 1,1, size(logw,1) ), [1,3,2] ) ) + logw';
tmp = permute(tmp, [1,3,2]);
max_global = max(tmp(:));

if num_post>0
%positive
max_post = max(tmp - (mask<0)*max_global,[],1);
res_post = reshape(log1p(sum(exp(tmp - max_post) .* (mask>0),1) -1) + max_post, size(x,2), size(logw,1));
end

if num_neg>0
%negative
max_neg = max(tmp - (mask>0)*max_global,[],1);
res_neg = reshape(log1p(sum(exp(tmp - max_neg) .* (mask<0),1) -1) + max_neg, size(x,2), size(logw,1));
end


if num_post>0 && num_neg>0
    [res, sig] = logsubexp(res_post',res_neg');
elseif num_post>0
    res = res_post';
    sig = ones( size(logw,1), size(x,2) );
else
    res = res_neg';
    sig = -ones( size(logw,1), size(x,2) );
end
assert( all(size(res) == size(ones( size(logw,1), size(x,2) ))) )
end
