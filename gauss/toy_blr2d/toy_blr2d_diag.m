addpath(genpath('.'))

clear all;
% Set the variance of prior (alpha=1/lambda)
alpha=100;


% ###################################################
% ############# Generate Synthetic Data #############
% ###################################################

dataset_name = 'murphy_synth';
[y, X, ytest, Xtest] = get_data_log_reg(dataset_name, 0);
t = (y+1) / 2;
D = 2;


% ##############################################################
% ############# Plot the Contours of the Posterior #############
% ##############################################################

Range = 30;
Step=0.5;
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

%% prior, likelihood, posterior
f=W*X';
Log_Prior = log(gaussProb(W, zeros(1,D), eye(D).*alpha));
Log_Like = W*X'*t - sum(log(1+exp(f)),2);
Log_Joint = Log_Like + Log_Prior;
post = exp(Log_Joint - logsumexpPMTK(Log_Joint(:),1));

% Identify the MAP estimate
[i,j]=max(Log_Joint);
wmap = W(j,:);

% #####################################################
% ############# Compute MF-Exact Solution #############
% #####################################################

% Compute the mf-exact solution.
D = 2;
sig = [1;1];
m = zeros(D,1);
v0 = [m; sig(:)];
funObj = @funObj_mfvi_exact;
optMinFunc = struct('display', 0, 'Method', 'lbfgs', 'DerivativeCheck', 'off','LS', 2, 'MaxIter', 200, 'MaxFunEvals', 200, 'TolFun', 1e-4, 'TolX', 1e-4);
gamma = ones(D,1)./alpha;
[v, f, exitflag, inform] = minFunc(funObj, v0, optMinFunc, t, X, gamma);
w_exact_vi = v(1:D);
U = v(D+1:end);
C_exact_vi = diag(U.^2);



% ###############################################################
% ############# Run VOGN, Vadam, and iBayesLRule ################
% ###############################################################

[N,D] = size(X);
maxIters = 500000;
colors = {'r','b','g','m'};

method = {'iBayesLRule', 'Vadam', 'VOGN'};
beta_momentum = 0.1;
grad_average = zeros(D,1);
w_all = zeros(2,maxIters,length(method));
Sigma_all = zeros(2,2,maxIters,length(method));

for m = 1:length(method)
   setSeed(1);
   name = method{m}
   % initialize
   w = [1, 1]';
   ss = 0.01;% alpha
   gamma = 0.99;% 1 - beta;

   beta_orig = 0.1
   ss_orig = 0.1
   for t = 1:maxIters
      if mod(t,10000)==0
          fprintf('Running at iter=%d\n', t);
      end

      %fprintf('%d) %.3f %.3f\n', t, w(1), w(2));
      M = 10;
      nSamples = 1;

      ss = ss_orig / (1 + t^(.55));
      beta = beta_orig / (1 + t^(.55));

      gamma = 1 - beta;
      switch name
      case 'VOGN'
          M = 1;
          if t == 1
              S_corrected = eye(D);
              S = S_corrected;
          end
      case {'Proj', 'Vadam'}
          if t == 1;
            S_corrected = eye(D);
            S = S_corrected;
        end

      case 'iBayesLRule'
          if t == 1;
            S_corrected = eye(D);
          end
      otherwise
          if t == 1; S = 100*eye(D); end
      end

      % draw minibatches
      i = unidrnd(N, [M,1]);

      % compute g and H
      g = 0; H = 0; H_rep =0;

      Prec = S_corrected + eye(D)./alpha;
      Sigma = inv(Prec);
      U = chol(Sigma); % upper triangle matrix
      L = chol(Prec, 'lower');
      %H_exact = 0;
      for k = 1:nSamples
         % for the k'th sample
         raw_noise = randn(D,1);
         epsilon = (L') \raw_noise;
         wt = w + epsilon;

         %The Hessian trick
         %[~,gpos,Hpos] = LogisticLoss(wt,X(i,:),y(i));
         %H_exact = H_exact + (Hpos/M)/nSamples;

         %The reparametrization trick
         [~,gpos] = LogisticLoss(wt,X(i,:),y(i));

         % gradient
         g = g + (gpos/M)/nSamples;

         switch name
         case {'iBayesLRule'}
             tmp = (L*raw_noise)*gpos';
             H_rep = H_rep + ( (tmp+tmp')/(2*M) )/ nSamples;
         case {'VOGN'}
             %Gauss-Newton
             [~,~,Gpos] = LogisticLossGN(wt,X(i,:),y(i,:));
             H = H + (Gpos/M)/nSamples;
         end
      end
      % unbiased estimate
      ghat = N*g;

      switch name
      case {'iBayesLRule'}
          Hhat = diag(diag(N*H_rep));
      case {'VOGN'}
          Hhat = N*diag(diag(H));
      case {'Vadam'}
          Hhat = diag(diag((ghat * ghat')  ./ N));
      end
      %Hhat = N*diag(diag(H_exact));

     switch name
     case {'Vadam'}
        grad_average = ((1-beta_momentum) .* grad_average) + ((ghat + (w ./ alpha)) .* beta_momentum);
        grad_average_corrected = grad_average ./ (1 - (1 - beta_momentum)^t);

        S = gamma * S + (1-gamma) * Hhat;
        S_corrected = S;
        w = w - ss*( ( (sqrt(S_corrected) + eye(D)./alpha)./ (1 - (gamma)^t) ) \ grad_average_corrected);

        w_all(:,t,m) = w(:);
        Sigma_all(:,:,t,m) = inv( S_corrected + eye(D)./alpha );

     case {'VOGN'}
        grad_average = ((1-beta_momentum) .* grad_average) + ((ghat + (w ./ alpha)) .* beta_momentum);
        grad_average_corrected = grad_average ./ (1 - (1 - beta_momentum)^t);

        S = gamma * S + (1-gamma) * Hhat;
        S_corrected = S;
        w = w - ss*( ( (S_corrected + eye(D)./alpha)./ (1 - (gamma)^t) ) \ grad_average_corrected);

        w_all(:,t,m) = w(:);
        Sigma_all(:,:,t,m) = inv( S_corrected + eye(D)./alpha );

     case 'iBayesLRule'
        grad_average = ((1-beta_momentum) .* grad_average) + ((ghat + (w ./ alpha)) .* beta_momentum);
        grad_average_corrected = grad_average ./ (1 - (1 - beta_momentum)^t);

        P = S_corrected + eye(D)./alpha;
        %w = w - ss*( ( P./ (1 - (gamma)^t) ) \ grad_average_corrected);
        w = w - ss*(1 - (gamma)^t)*( L'\(L\grad_average_corrected) );

        g_V = Hhat + eye(D)./alpha - P;
        %P2 = P + (1-gamma) * g_V + (1-gamma)^2/2*(g_V*(P\g_V))
        half = L' + (1-gamma)*(L\g_V);
        P = (P + half'*half)/2;
        S_corrected  = P - eye(D)./alpha;

        w_all(:,t,m) = w(:);
        Sigma_all(:,:,t,m) = inv( S_corrected + eye(D)./alpha );
     otherwise
         error('no such method');
     end
   end
end

% save the experiment.
file_name = strcat('./toy_example_experiment_data.mat');
save(file_name, 'maxIters',  'method', 'dataset_name', 'Sigma_all', 'w_all', 'w1', 'w2', 'W', 'f', 'Log_Prior', 'Log_Like', 'Log_Joint', 'post', 'wmap', 'w_exact_vi', 'C_exact_vi');


% ######################################################
% ############# Plot and Save Figure 2 (a) #############
% ######################################################

make_toy_blr2d_diag
