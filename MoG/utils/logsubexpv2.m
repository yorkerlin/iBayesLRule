function [res, sind]=logsubexpv2(a,b)
assert ( all( size(a) == size(b) ) )
ta=a(:)'; tb=b(:)';
tmp=[ta;tb];
sind = sign(ta-tb);
res = zeros( size(sind) );

res( sind ~= 0 ) = logsubexp_Helper( tmp(:,sind~=0) );

sind=reshape(sind, size(a));
res =reshape(res, size(a));

end

function res=logsubexp_Helper(tmp)
tv=max(tmp,[],1);
res=tv+log1p(-exp(min(tmp,[],1)-tv));
end

