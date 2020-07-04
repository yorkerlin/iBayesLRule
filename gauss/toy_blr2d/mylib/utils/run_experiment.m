% @Author: amishkin
% @Date:   2018-07-10T13:52:49-07:00
% @Email:  amishkin@cs.ubc.ca
% @Last modified by:   aaronmishkin
% @Last modified time: 2018-07-25T13:28:35-07:00

% ##########################################################
% ############# Helper for Running Expeirments #############
% ##########################################################

function [] = run_experiment(dataset_name, method, M_list, epoch_list, alpha, beta, decay_rate, num_samples, num_restarts, mu_start, sigma_start, random_split, output_dir)
    for s = 1:num_restarts
        for m = 1:length(M_list)
            % Set the random seed to be the restart number.
            if random_split
                [y, X, y_te, X_te] = get_data_log_reg(dataset_name, s);
            else
                [y, X, y_te, X_te] = get_data_log_reg(dataset_name, 1);
                % The seed is set in get_data_log_reg, so we must reset it after the function call.
                setSeed(s);
            end
            [N,D] = size(X);

            mini_batch_size = M_list(m);
            num_epochs = epoch_list(m);

            save_name = strcat(dataset_name, '_', method, '_M_', num2str(mini_batch_size), '_restart_', num2str(s), '.mat');
            trial_params = [num_samples, mini_batch_size, num_epochs, beta, alpha, decay_rate];
            parameter_string = sprintf('S: %0.5g, M: %0.5g, Epochs: %0.5g, beta: %0.5g, alpha: %0.5g, decay rate: %0.5g', trial_params)

            [delta, parameter_object] =  get_expt_params(dataset_name, method, trial_params);
            deltas = [1e-5; delta.*ones(D-1,1)]; % prior variance

            % Run the correct method:
            if strcmp(method, 'mf_exact')
                % mf_exact is uses L-BFGS for optimization and so is quite different from Vadam and VOGN --- we place it in it's own file for this reason.
                [nlZ, log_loss, Sigma, mu] = mf_exact(method, y, X, deltas, y_te, X_te, parameter_object, mu_start, sigma_start);
                log_loss
            else
                [nlZ, log_loss, Sigma, mu] = my_methods(method, y, X, deltas, y_te, X_te, parameter_object, mu_start, sigma_start);
            end

            if isfield(parameter_object,'mini_batch_size')
                mini_bsz = parameter_object.mini_batch_size;
            else
                mini_bsz = N;
            end
            ipp = floor(N / mini_bsz);
            % Print information about the most recent run:
            fprintf('%s Restart Number: %.4f, ELBO: %.4f, LogLoss: %.4f\n', method, s, nlZ, log_loss);
            file_name = strcat(output_dir, '/' , save_name);
            save(file_name, 'method', 'dataset_name', 'trial_params', 'Sigma', 'mu', 'ipp', 'log_loss', 'nlZ');
        end
    end
end
