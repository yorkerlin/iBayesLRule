function [infos] = blinreg(name,option,method,maxIters,ss,B)
[delta, sig2] = get_expt_params(name);

infos = zeros(maxIters,2);

seed = 0;
[y, X, y_te, X_te] = get_data_lin_reg(name, seed);
size(X)
X = X';
X_te = X_te';
[d,n] = size(X);
m_0 = zeros(d,1);
P_0 = delta*eye(d);
L_prec_0 = chol(P_0,'lower');

m = m_0;
P = P_0;
L_prec = chol(P,'lower');

[exact_m,exact_P] = get_exact(X,y,m_0,P_0,sig2);
exact_L_prec = chol(exact_P,'lower');
M = ceil(n/20);
decay = 0.8;

[exact_pri_obj,~,~] = log_gauss_prior(exact_m,exact_P,exact_L_prec,m_0,P_0,L_prec_0);
[exact_ent_obj,~,~] = gauss_entropy(exact_m,exact_P,exact_L_prec);
[exact_lik_obj] = log_gauss_lik_exact(y,X,sig2,exact_m,exact_L_prec,n,n,0);
exact_obj = exact_pri_obj + exact_ent_obj + exact_lik_obj;

fname = sprintf('./blinreg-%s-exact.mat', name)
save(fname, 'exact_obj')

switch method
case {'bbvi'}
    new_param = 0;
    epsilon=1e-8; decay_factor_mean=0.9; decay_factor_var=0.999;
    g_mean_w = zeros(d,1);
    g_mean_C = zeros(d,d);

    g_var_w = zeros(d,1);
    g_var_C = zeros(d,d);
    C = chol(P\eye(d),'lower');
end


for iter = 1:maxIters
    lr = ss;
    raw_noise = randn(d, B);
    z = m + (L_prec')\raw_noise;

    [pri_obj,pri_grad_m,pri_grad_V] = log_gauss_prior(m,P,L_prec,m_0,P_0,L_prec_0);
    [ent_obj,ent_grad_m, ent_grad_V] = gauss_entropy(m,P,L_prec);

    [lik_obj_mc,grad_z,hess_z] = log_gauss_lik(y,X,sig2,z,n,M,iter);
    %[lik_obj_mc,grad_z,hess_z] = log_gauss_lik2(y,X,sig2,z,n,M,iter);

    [lik_obj] = log_gauss_lik_exact(y,X,sig2,m,L_prec,n,n,iter);
    obj = pri_obj + ent_obj + lik_obj;

    kl =gauss_kl(m,L_prec,exact_m,exact_L_prec);
    fprintf('%08d) obj=%.6f obj-diff=%.8f kl=%.6f\n', iter-1, -obj, exact_obj-obj, kl);
    infos(iter,1) = -obj;
    infos(iter,2) = kl;

    switch option
    case 'hess' %the Hessian trick
        lik_grad_V = mean(hess_z,3)/2.;
    case 'first'
        switch method
        case 'vogn'
            setSeed(iter);
            ind = randperm(n,M)';
            g_lik = zeros(d,B);
            gSigma_lik = zeros(d,d);
            for i=1:M
                [~,g2] = log_gauss_lik(y(ind(i)),X(:,ind(i)),sig2,z,1,1,iter);
                tmp = einsum('kj,ij->ki', g2, g2);
                gSigma_lik = gSigma_lik + tmp/B;
                g_lik = g_lik + g2/M;
            end
            grad_z2 = g_lik*n;
            lik_grad_V = -(gSigma_lik)*(n/(2*M));
        otherwise %the re-parametrization trick
            tmp=einsum('ij,kj->ik',L_prec*raw_noise,grad_z);
            H_est = (tmp +tmp')/(2.0*B);
            lik_grad_V = H_est/2.;
        end
    otherwise
        error('unknown estimation')
    end

    lik_grad_m = mean(grad_z,2);
    dV = pri_grad_V + ent_grad_V + lik_grad_V;
    dm = pri_grad_m + ent_grad_m + lik_grad_m;

    switch method
    case {'cvi','vogn'}
        while 1
            try
                chol(P - (2.0*lr)*dV);
            catch
                disp('h')
                lr = lr*decay;
                continue;
        end
            break
        end
        P = P - 2.0*lr*dV;
        m = m + lr*(P\dm);
    case 'iBayesLRule'
        m = m + lr*(L_prec'\(L_prec\dm));
        %m = m + lr*(P\dm); % m = m + beta P\(gMu);
        right = 2.0*lr*(L_prec\dV)-L_prec';
        P = (P+ right'*right)/2;
        %P = P - (2.0*lr)*dV + (lr*lr*2.0)*(dV*(P\dV)); % P = P - 2*lr*dV + lr*lr/2*((2*dV)*P\(2*dV));
    case 'bbvi'
        dC = tril(dV*(2*C)); % V= C*C'
        [g_m,g_mean_w,g_var_w] = adam(dm,g_mean_w,g_var_w,lr,iter,decay_factor_mean,decay_factor_var,epsilon);
        m = m + g_m;
        [g_C,g_mean_C,g_var_C] = adam(dC,g_mean_C,g_var_C,lr,iter,decay_factor_mean,decay_factor_var,epsilon);
        C = C + g_C;
        P = C'\(C\eye(d));
    otherwise
        error('unknown method')
    end
    try
        L_prec = chol(P,'lower');
    catch
        infos(iter+1,1) = -1;
        infos(iter+1,2) = -1;
        disp('break')
        break
    end
end

end



function [exact_m,exact_P] = get_exact(X,y,m_0,P_0,sig2)
exact_P = P_0 + X*X'/sig2;
exact_m = exact_P\( X*y/sig2 + P_0*m_0 );
end


function [delta, sig2] = get_expt_params(dataset_name)
switch dataset_name
case {'abalone_scale'}
    sig2 = 162.9751;
    delta =0.0046;
otherwise
  error('Unknown dataset name')
end
end

function [lp] = log_gauss_lik_exact(y,X,sig2,mu,L_prec,n,m,iter)
if n==m
    ind = [1:n];
else
    setSeed(iter);
    ind = randperm(n,m)';
end
assert(length(ind) == m)
X = full(X(:,ind));
y = y(ind);
assert( size(y,1) == size(X,2) )
assert( size(y,2) == 1)

XTmu=X'*mu;
left = L_prec\X;
yminusXTmu = y - XTmu;


lp = ( -log(2.0*pi) - log(sig2) -  ( mean(yminusXTmu.^2, 1) +  sum(sum(left.^2))/m )/sig2 )*(n/2.);
end

function [varargout] = log_gauss_lik(y,X,sig2,z,n,m,iter)
%y is m by 1, where m is the size of a minibatch
%X is d by m, where d is the dim of features
%sig2 is the likelihood variance
%z is d by B, where B is # of MC samples
if n==m
    ind = [1:n];
else
    setSeed(iter);
    ind = randperm(n,m)';
end
assert(length(ind) == m)
X = full(X(:,ind));
y = y(ind);
assert( size(y,1) == size(X,2) )
assert( size(y,2) == 1)
assert( size(X,1) == size(z,1) )
yminusXTz = y - X'*z;
lp = ( -log(2.0*pi) - log(sig2) -  mean(yminusXTz.^2, 1)/sig2 )*(n/2.);
lp = sum(lp);
if nargout>1
    grad_z = X*yminusXTz/sig2*(n*1.0/size(y,1));
    if nargout>2
        hess_z = (X*X')*(-n/(sig2*size(y,1)));
        hess_z = repmat(hess_z,1,1, size(z,2) );
        varargout = {lp,grad_z,hess_z};
    else
        varargout = {lp, grad_z};
    end
else
    varargout = {lp};
end
end


function [varargout] = log_gauss_lik2(y,X,sig2,z,n,m,iter)
    if n==m
        ind = [1:n];
    else
        setSeed(iter);
        ind = randperm(n,m)';
    end
    assert(length(ind) == m)
    X = full(X(:,ind));
    y = y(ind);
    assert( size(y,1) == size(X,2) )
    assert( size(y,2) == 1)
    assert( size(X,1) == size(z,1) )
    %X is d by m, where d is the dim of features
    %z is d by B, where B is # of MC samples
    %y is m by 1, where m is the size of a minibatch
    %sig2 is the variance parameter of T

    X_hat = X'; %m by d
    betaDraws = z; %d by B
    yhat = y; %m by 1
    B = size(betaDraws, 2);

    fn = (X_hat*betaDraws)';%B by m
    hyp = [log(sig2)/2.];
    lik = {@likGauss};

    % compute MC approximation (code taken from GPML)
    yhat = repmat(yhat(:)', B, 1);
    if nargout==1
        f = feval(lik{:}, hyp, yhat, fn, [], 'infLaplace');
    else
        [f, df, d2f] = feval(lik{:}, hyp, yhat, fn, [], 'infLaplace');
    end
    assert( size(f',1) == m )
    lp = mean(f',1)*n; %lp =  sum_x \log link(x,w)
    assert( length(lp) == size(z,2) )
    lp = sum(lp);

    if nargout>1
        gm = df';
        grad = X_hat'*gm*(n*1.0/m);  %G
        if nargout>2
            gv = d2f'/2;
            tmp1 = repmat(X_hat,1,1,B) .* reshape(2*gv,size(gv,1),1,size(gv,2));%diag(2*gv)*X
            hess = einsum('ij,jkp->ikp',X_hat',tmp1)*(n*1.0/m);
            varargout = {lp,grad,hess};
        else
            varargout = {lp, grad};
        end
    else
        varargout = {lp};
    end
end

function [ll] =gauss_kl(m,L,m_0,L_0)
% P is the precision matrix
%L_0 = chol(P_0,'lower');%P_0 = L_0*L_0'
half= L_0'*(m-m_0);
%L = chol(P,'lower');
left = L\L_0;
%ll = ((m-m_0)'*(P_0*(m-m_0)) - size(P,1) - logdet(P_0) + logdet(P) + trace(P_0\P))/2.0;
ll = ( sum(half.^2)- size(L,1) + sum(sum(left.^2)) )/2.0 - sum(log(diag(L_0))) +  sum(log(diag(L)));
ll = abs(ll);
end

function [varargout]= log_gauss_prior(m,P,L_prec,m_0,P_0,L_prec_0)
%m is the mean
%P is the precision matrix
assert( size(m_0,2) == 1 )
assert( size(m_0,1) == size(P_0,1) )
assert( all(size(m)==size(m_0)) )
assert( all(size(P)==size(P_0)) )
%P = L_prec*L_prec'
%P_0 = L_prec_0*L_prec_0'
tmp = L_prec\L_prec_0;
tmp = tmp.*tmp;
%sum(tmp(:)) == trace(inv(P)*P_0)
lp = ( 2.0*sum(log(diag(L_prec_0))) - log(2*pi)*size(P_0,1) - (m-m_0)'*P_0*(m-m_0) - sum(tmp(:)) )/2.0;
lp = sum(lp);
if nargout>1
    grad_m = -P_0*(m-m_0);
    if nargout>2
        grad_V = -P_0/2.;
        varargout = {lp,grad_m,grad_V};
    else
        varargout = {lp,grad_m};
    end
else
    varargout = {lp};
end
end

function [ll, grad_m, grad_V] = gauss_entropy(m,P,L_prec)
%m is the mean
%P is the precision matrix (V=inv(P))
ll = ( ( log(2*pi)+1 )*size(P,1) - 2.0*sum(log(diag(L_prec))) )/2.0;
grad_m = 0*m;
grad_V = P/2.0;
end
