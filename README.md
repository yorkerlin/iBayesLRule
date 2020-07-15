Code for ICML 2020 paper on [Handling the Positive-Definite Constraint in the Bayesian Learning Rule](https://arxiv.org/abs/2002.10060)
<br />  
We propose an Efficient Riemannian/Natural Gradient Variational Inference method
* To-do List:
  * Added [slides](https://github.com/yorkerlin/iBayesLRule/blob/master/slides.pdf) of the talk 
  * Added a [link](https://www.youtube.com/watch?v=nu1hT-LExFg) of the ICML talk
  * Added a Python [implementatoin](https://github.com/yorkerlin/iBayesLRule/tree/master/inverse_gauss/) of the implicit reparameterization gradient for inverse Gaussian distribution (See Appendix H.1)
  * Added a Matlab [implementation](https://github.com/yorkerlin/iBayesLRule/tree/master/gauss) about Gaussian approximation (See Appendix E) 
 ![](./plots/blr2d_full.png) 
  * Added a Matlab [implementation](https://github.com/yorkerlin/iBayesLRule/tree/master/MoG) about  MoG approximation (See Appendix J)
 ![](./plots/MOG.png) <br />  
 examples of MoG approximation:
 [star](https://github.com/yorkerlin/iBayesLRule/blob/master/MoG/star.m), [double-banana](https://github.com/yorkerlin/iBayesLRule/blob/master/MoG/doublebanana.m), [Laplace](https://github.com/yorkerlin/iBayesLRule/blob/master/MoG/laplace.m), [BNN](https://github.com/yorkerlin/iBayesLRule/blob/master/MoG/bnn.m), [mixture of Ts](https://github.com/yorkerlin/iBayesLRule/blob/master/MoG/mixT_demo20d.m)
  * To add a Matlab [implementation](https://github.com/yorkerlin/iBayesLRule/) about Gamma approximation (See Appendix F)
  * To add a Python [implementation](https://github.com/yorkerlin/iBayesLRule/) for the Adam-like update using factorized/diagonal Gaussian approximation (See Appendix E.3)

