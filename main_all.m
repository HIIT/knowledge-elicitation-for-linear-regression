close all
clear all
RNG_SEED = rng;

%% Parameters and Simulator setup 
MODE = 1; 
% MODE specifies the  type of feedback and the model that we are using
%           0: Feedback on weight values. Model: Gaussian prior 
%           1: Feedback on weight values. Model: spike and slab prior
%           2: Feedback on relevance of features. Model: spike and slab prior

%data parameters for simulation data
num_features             = 100; %[start,step,max] This can be a set of values (e.g. 1:100) or just one value (e.g. 100)
num_trainingdata         = 2:1:100; %[start,step,max] This can be a set of values (e.g. 1:10:500) or just one value (e.g. 5)
num_userdata             = 500; %data that will be used in active learning
max_num_nonzero_features = 10; % maximum number of features that are nonzero --- AKA sparsity measure

%Algorithm parameters
num_iterations = 50; %total number of user feedback
num_runs       = 100; %total number of runs (necessary for averaging results) 

%model parameters
normalization_method = 3; %normalization method for generating the data (Xs)
model_params   = struct('Nu_y',1, 'Nu_theta', 1, 'Nu_user', 0.1, 'P_user', 0.95, 'simulated_data', 1);
sparse_options = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0, 'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
sparse_params  = struct('sigma2',model_params.Nu_y^2, 'tau2', model_params.Nu_theta^2 ,'eta2',model_params.Nu_user^2,'p_u', model_params.P_user);
%% METHOD LIST
% Set the desirable methods to 'True' and others to 'False'. only the 'True' methods will be considered in the simulation
METHODS_ED = {     
     'False',  'Max(90% UCB,90% LCB)'; 
     'True',  'Uniformly random';
     'False', 'random on the relevelant features';
     'False', 'max variance';
     'False', 'Bayes experiment design';
     'False',  'Expected information gain';
     'False', 'Bayes experiment design (tr.ref)';
     'True',  'Expected information gain (post_pred)';
     'False',  'Expected information gain (post_pred), non-sequential';
     'False',  'Expected information gain (post_pred), fast approx'; %Only available for MODE = 2?
     'False',  'Expected information gain (post_pred), fast approx, non-sequential' %Only available for MODE = 2?
     };
METHODS_AL = {
     'False',  'AL:Uniformly random';
     'False',  'AL: Expected information gain'
     }; 
METHODS_GT = { %these are non-sequential methods
     'False',  'Ground truth - all data';
     'False',  'Ground truth - all feedback'
     }; 
Method_list_ED = [];
for m = 1:size(METHODS_ED,1)
    if strcmp(METHODS_ED(m,1),'True')
        Method_list_ED = [Method_list_ED,METHODS_ED(m,2)];
    end
end
Method_list_AL = [];
for m = 1:size(METHODS_AL,1)
    if strcmp(METHODS_AL(m,1),'True')
        Method_list_AL = [Method_list_AL,METHODS_AL(m,2)];
    end
end
Method_list_GT = [];
for m = 1:size(METHODS_GT,1)
    if strcmp(METHODS_GT(m,1),'True')
        Method_list_GT = [Method_list_GT,METHODS_GT(m,2)];
    end
end
Method_list = [Method_list_GT, Method_list_ED, Method_list_AL];
num_methods = size(Method_list,2); %number of decision making methods that we want to consider
%% Main
Loss_1 = zeros(num_methods, num_iterations, num_runs, size(num_features,2),size(num_trainingdata,2));
Loss_2 = zeros(num_methods, num_iterations, num_runs, size(num_features,2),size(num_trainingdata,2));
Loss_3 = zeros(num_methods, num_iterations, num_runs, size(num_features,2),size(num_trainingdata,2));
decisions = zeros(num_methods, num_iterations, num_runs, size(num_features,2),size(num_trainingdata,2));

for n_f = 1:size(num_features,2); 
    n_f
    sparse_params.rho = max_num_nonzero_features/num_features(n_f);
    model_params.P_zero = sparse_params.rho;
    for n_t = 1:size(num_trainingdata,2);
        n_t
        num_data = 500 + num_trainingdata(n_t) + num_userdata; % total number of data (training and test)
        tic
        for run = 1:num_runs
            disp(['run number ', num2str(run), ' from ', num2str(num_runs), '. acc time = ', num2str(toc) ]);
            num_nonzero_features = min( num_features(n_f), max_num_nonzero_features);
            %Theta_star is the true value of the unknown weight vector
            % non-zero elements of theta_star are generated based on the model parameters
            theta_star = model_params.Nu_theta*randn( num_nonzero_features, 1); 
            theta_star = [theta_star; zeros(num_features(n_f)-num_nonzero_features,1)]; % make it sparse
            z_star = theta_star ~= 0; % the true value for the latent variable Z in spike and slab model
            %generate new data for each run (because the results is sensitive to the covariate values)
            X_all   = generate_data(num_data,num_features(n_f), normalization_method);
            Y_all   = normrnd(X_all*theta_star, model_params.Nu_y);
            [X_train, X_user, X_test, Y_train, Y_user, Y_test] = partition_data(X_all, Y_all, num_userdata, num_trainingdata(n_t));
            %% main algorithms (ED, AL, and GT)
            for method_num = 1:num_methods
                method_name = Method_list(method_num);
                %Feedback = values (1st column) and indices (2nd column) of user feedback
                Feedback = [];            %only used in experimental design methdos
                %selected_data = indices of data selected by active learning from X_user and Y_user
                selected_data = [];       %only used in active learning methods
                sparse_options.si = [];   %carry prior site terms between interactions
                %% Calculate ground truth solutions
                if find(strcmp(Method_list_GT, method_name))
                    if find(strcmp('Ground truth - all data', method_name))
                        %calculate the posterior based on all train+user data
                        posterior = calculate_posterior([X_train, X_user], [Y_train; Y_user], Feedback, ...
                            model_params, MODE, sparse_params, sparse_options);
                    end
                    if find(strcmp('Ground truth - all feedback', method_name))
                        %calculate the posterior based on all feedbacks
                        for feature_index = 1:size(X_train,1)
                            new_fb_value = user_feedback(feature_index, theta_star, z_star, MODE, model_params);
                            Feedback = [Feedback; new_fb_value , feature_index];
                        end
                        posterior = calculate_posterior(X_train, Y_train, Feedback, ...
                            model_params, MODE, sparse_params, sparse_options);
                    end
                    Y_hat = X_test'*posterior.mean;
                    [mse,log_pp_test,log_pp_train] = calculate_loss(X_train,Y_train, posterior, ...
                        X_test, Y_hat, Y_test, model_params);
                    Loss_1(method_num, :, run, n_f ,n_t) = mse;
                    Loss_2(method_num, :, run, n_f ,n_t) = log_pp_test;
                    Loss_3(method_num, :, run, n_f ,n_t) = log_pp_train;
                    continue
                end   
                %% for non-sequential ED methods find the suggested queries before user interaction
                if strfind(char(method_name),'non-sequential')
                    posterior = calculate_posterior(X_train, Y_train, [], model_params, MODE, sparse_params, sparse_options);
                    %find non-sequential order of features to be queried from the user
                    non_seq_feature_indices = decision_policy(posterior, method_name, z_star, X_train, Y_train, ...
                        [], model_params, MODE, sparse_params, sparse_options);
                end
                %% User interaction
                for it = 1:num_iterations %number of user feedback
                    %calculate the posterior based on training + feedback until now
                    posterior = calculate_posterior([X_train, X_user(:,selected_data)], [Y_train; Y_user(selected_data)], Feedback, ...
                        model_params, MODE, sparse_params, sparse_options);
                    sparse_options.si = posterior.si;
                    %% calculate different loss functions
                    Y_hat = X_test'*posterior.mean;
                    [mse,log_pp_test,log_pp_train] = calculate_loss(X_train,Y_train, posterior, ...
                        X_test, Y_hat, Y_test, model_params);
                    Loss_1(method_num, it, run, n_f ,n_t) = mse;
                    Loss_2(method_num, it, run, n_f ,n_t) = log_pp_test;
                    Loss_3(method_num, it, run, n_f ,n_t) = log_pp_train;
                    %% If ED: make a decision based on ED decision policy
                    if find(strcmp(Method_list_ED, method_name))
                        %for non-sequential methods, use the saved order
                        if strfind(char(method_name),'non-sequential')
                            feature_index = non_seq_feature_indices(it);
                        else
                            %for sequential methods find the next decision based on feedback until now
                            feature_index = decision_policy(posterior, method_name, z_star, X_train, Y_train, ...
                                Feedback, model_params, MODE, sparse_params, sparse_options);
                        end
                        decisions(method_num, it, run, n_f ,n_t) = feature_index;
                        %simulate user feedback
                        new_fb_value = user_feedback(feature_index, theta_star, z_star, MODE, model_params);
                        Feedback = [Feedback; new_fb_value , feature_index];
                    end
                    %% If AL: add a new data point based on AL decision policy
                    if find(strcmp(Method_list_AL, method_name))
                        [new_selected_data] = decision_policy_AL(posterior, method_name, ...
                            [X_train, X_user(:,selected_data)] , [Y_train; Y_user(selected_data)], ...
                            X_user, selected_data, model_params, sparse_params, sparse_options);
                        selected_data = [selected_data;new_selected_data];
                    end
                end
            end
        end
    end
end

%% averaging and plotting
save('results_all', 'Loss_1', 'Loss_2', 'Loss_3', 'decisions', 'model_params', 'sparse_options', 'sparse_params', ...
    'z_star', 'Method_list', 'num_features','num_trainingdata', 'MODE', 'normalization_method', 'RNG_SEED')
evaluate_results_all
