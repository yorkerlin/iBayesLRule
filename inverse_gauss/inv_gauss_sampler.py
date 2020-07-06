from autograd.extend import primitive, defvjp
import numpy.random as npr
import autograd.numpy as np
from autograd import grad, value_and_grad
import autograd.scipy as scipy
import autograd.scipy.special as special
from scipy import special as org_special
from scipy.stats import norm
from autograd.tracer import getval
from scipy.special import logsumexp

def log_standard_gauss(x): #norm.logpdf(x)
    return -( np.log(2.0*np.pi)+(x**2) )/2.0

def dalpha(z,alpha,beta):
    #see d_z/d_alpha at Appendix H.1 (page 27) of https://arxiv.org/pdf/2002.10060v8.pdf
    tmp = -( np.sqrt(alpha*z)*beta + np.sqrt(alpha/z) )
    return z/alpha - 2.0*beta*np.exp( -np.log(alpha)/2.0 + np.log(z)*(3.0/2.0) + log_gauss_cdf(tmp)  - log_standard_gauss(tmp) )

def dbeta(z,alpha,beta):
    #see d_z/d_beta at Appendix H.1 (page 27) of https://arxiv.org/pdf/2002.10060v8.pdf
    tmp = -( np.sqrt(alpha*z)*beta + np.sqrt(alpha/z) )
    return -np.exp(np.log(alpha)/2.0 + np.log(z)*(3.0/2.0) + log_gauss_cdf(tmp)  - log_standard_gauss(tmp))*2.0

@primitive
def log_gauss_cdf(z): #norm.logcdf(x)
    return org_special.log_ndtr(z)

def log_gauss_cdf_vjp(ans, z):
    return lambda g: g * np.exp(log_standard_gauss(z) - org_special.log_ndtr(z))

defvjp(log_gauss_cdf,
       log_gauss_cdf_vjp,
       None)

def inv_gauss_sampler(alpha, beta, sampler):
    assert( np.all(alpha>0.) )
    assert( np.all(beta>0.) )
    #under the parametrization defined at the Wikipedia page (https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution),
    #we have lambda = alpha, mu = 1/beta
    return sampler.wald(mean=1.0/beta, scale=alpha)

@primitive
def myinv_gauss(alpha,beta,sampler):
    return inv_gauss_sampler(alpha, beta, sampler)

def myinv_gauss_alpha_vjp(ans,alpha,beta,sampler):
    # tmp = -( np.sqrt(alpha*ans)*beta + np.sqrt(alpha/ans) )
    # grad_alpha=ans/alpha - 2.0*beta*np.exp( -np.log(alpha)/2.0 + np.log(ans)*(3.0/2.0) + log_gauss_cdf(tmp) - log_standard_gauss(tmp) )
    grad_alpha=dalpha(ans,alpha,beta)
    def build(g):
        return g * grad_alpha
    return build

def myinv_gauss_beta_vjp(ans,alpha,beta,sampler):
    # tmp = -( np.sqrt(alpha*ans)*beta + np.sqrt(alpha/ans) )
    # grad_beta=-np.exp(np.log(alpha)/2.0 + np.log(ans)*(3.0/2.0) + log_gauss_cdf(tmp) - log_standard_gauss(tmp))*2.0
    grad_beta = dbeta(ans,alpha,beta)
    def build(g):
        return g * grad_beta
    return build

defvjp(myinv_gauss,
       myinv_gauss_alpha_vjp,
       myinv_gauss_beta_vjp)

def mylog_exp1(x):
    #a numerically stable version of the log exponential integral.
    #see Appendix H.1 (page 27) of https://arxiv.org/pdf/2002.10060v8.pdf
    assert(x>0)
    if x<100:
        return np.log( org_special.exp1(x) )

    #using the Asymptotic expansion
    #see Eq 3 at https://www.sciencedirect.com/science/article/pii/S0022169497001340
    tmp = np.cumsum( np.log(np.arange(int(x))+1) ) - (np.arange(int(x))+1)*np.log(x)
    neg = logsumexp(tmp[::2])  #odd index
    pos = logsumexp(tmp[1::2]) #even index
    return np.log1p(np.exp(pos)-np.exp(neg))-x-np.log(x)

#################################### examples are given below ##############################

def myx(param, sampler):
    alpha = param[0]
    beta = param[1]
    return np.sum( myinv_gauss(alpha,beta,sampler) )

def myinv_x(param, sampler):
    alpha = param[0]
    beta = param[1]
    return np.sum( 1.0/myinv_gauss(alpha,beta,sampler) )

def neg_logq(param, z):
    alpha = param[0]
    beta = param[1]
    lp = -z*alpha*beta*beta/2. - alpha/(2.*z) + np.log(alpha)/2. + alpha*beta - np.log(np.pi*2.)/2. - 3.0*np.log(z)/2.
    return -lp

def entropy(param,sampler):
    alpha = param[0]
    beta = param[1]
    return np.sum( neg_logq( getval(param), myinv_gauss(alpha,beta,sampler) ) )


fx = lambda param: myx(param, npr.RandomState(0))
finv_x = lambda param: myinv_x(param, npr.RandomState(0))
fent_x = lambda param: entropy(param, npr.RandomState(0))

gx = value_and_grad(fx)
ginvx = value_and_grad(finv_x)
gentropy = value_and_grad(fent_x)

d = 500 #use d MC samples (set d to be large so that we can see the MC approximation gives unbaised estimation)
print('Using %d MC samples'% d)
alpha = 2.0*np.ones((d,1))
beta = 4.0*np.ones((d,1))

sampler = npr.RandomState(0)
x = sampler.wald(mean=1.0/beta, scale=alpha)

param =(alpha, beta)

print('===========')
print('test case 1')
print('name:, exact, MC_estmation')
(res1, g1) = gx(param) # E[x] = 1/beta
print ('E[x]=1/beta:', 1.0/beta[0], res1/d)
print('d_E[x] / d_alpha:', 0, np.mean(g1[0]) ) #grad_alpha
print('d_E[x] / d_beta:', -1.0/(beta[0]**2), np.mean(g1[1]) ) #grad_beta


print('===========')
print('test case 2')
print('name:, exact, MC_estmation')
(res2, g2) = ginvx(param) # E[1/x] = beta + 1/alpha
print ('E[1/x]=beta+1/alpha:', beta[0]+1.0/alpha[0], res2/d)
print('d_E[1/x] / d_alpha:', -1.0/(alpha[0]**2), np.mean(g2[0]) ) #grad_alpha
print('d_E[1/x] / d_beta:', 1.0, np.mean(g2[1]) ) #grad_beta


print('===========')
print('test case 3')
print('name:, exact, MC_estmation')
(res3, g3) = gentropy(param) # E[-log q(x)] see Appendix H.1 (page 27) of https://arxiv.org/pdf/2002.10060v8.pdf
# tmp_not_stable  = org_special.exp1(2.0*beta[0]*alpha[0])*np.exp(2.0*beta[0]*alpha[0]) #not stable when beta[0]*alpha[0]>360
tmp = np.exp( mylog_exp1(2.0*beta[0]*alpha[0]) + 2.0*beta[0]*alpha[0])
print ('entropy:',  (-np.log(alpha[0]) - (np.log(beta[0])+tmp)*3.0  + 1 + np.log(np.pi*2.0) ) /2.0 , res3/d)
print('d_entropy / d_alpha:', 1.0/(alpha[0])-beta[0]*tmp*3.0,  np.mean(g3[0])  ) #grad_alpha
print('d_entropy / d_beta:', -( alpha[0]*tmp )*3.0,  np.mean(g3[1])  )   #grad_beta

