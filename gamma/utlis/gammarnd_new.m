function [x, dfda, dfdb] = gammarnd_new(a,b,D,idx,allow_zero)
   if nargin>3
        py.numpy.random.seed(int32(idx))
   end
   if nargin<5
       allow_zero = 0;
   end
   assert( size(a,1) == D )
   assert( size(a,1) > 0 )
   assert( size(a,2) == 1 )
   assert( size(b,2) == 1 )
   assert( all( size(a) == size(b) ) )
   if allow_zero ==0
       assert(min(a)>=1e-2)
   end
   var_gamma = py.numpy.random.gamma(a',1.); %have to use this
   var_gamma = double(py.array.array('d',py.numpy.nditer(var_gamma)));
   var_gamma = var_gamma';
   assert( all( size(a) == size(var_gamma) ) )
   x = var_gamma./b;

   if allow_zero ==0
       idx = (x==0);
       count_zero = sum(idx);
       if count_zero>0
           x(idx) = gammarnd_new(a(idx),b(idx),count_zero);
       end
       assert( sum(x>0) ==size(a,1) )
   end

   dfda = gammaGrad(a,x.*b)./b;
   dfdb = -x./b;
end
