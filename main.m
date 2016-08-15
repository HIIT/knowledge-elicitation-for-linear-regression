close all
clear all
%%TODO: 
% Real data (GDSC) is very strange. The code is not good for it yet.
% profile on
%% Parameters and Simulator setup
MODE                     = 2; 
% MODE specifies the  type of feedback and the model that we are using
%           0: Feedback on weight values. Model: Gaussian prior 
%           1: Feedback on weight values. Model: spike and slab prior
%           2: Feedback on relevance of features. Model: spike and slab prior

DATA_SOURCE = 'SIMULATION_DATA'; %use simulated data
% DATA_SOURCE = 'GDSC_DATA'; %Genomics of Drug Sensitivity in Cancer (GDSC)(not working)

%data parameters for SIMULATION_DATA
num_features         = 100; % total number of features
num_trainingdata     = 5; % number of samples (patients with available drug response)
num_nonzero_features = 10; % features that are nonzero
% One way to measure to check the method is to fix the following ration: #num_traingdata/num_features

%Algorithm parameters
num_iterations = 100; %total number of user feedback
num_runs       = 100;  %total number of runs (necessary for averaging results)
num_data       = 500 + num_trainingdata; % total number of data (training and test) - this is not important

%model parameters
model_params   = struct('Nu_y',0.1, 'Nu_theta', 1, 'Nu_user', 0.1, 'P_user', 0.99, 'P_zero', num_nonzero_features/num_features);
normalization_method = 1; %normalization method for generating the data (Xs)
sparse_options = struct('damp',0.5, 'damp_decay',1, 'robust_updates',2, 'verbosity',0, 'max_iter',100, 'threshold',1e-5, 'min_site_prec',1e-6);
sparse_params  = struct('sigma2',model_params.Nu_y^2, 'tau2', model_params.Nu_theta^2 ,'eta2',model_params.Nu_user^2,'p_u', model_params.P_user);
sparse_params.rho = model_params.P_zero;
%% METHOD LIST
% Set the desirable methods to 'True' and others to 'False'. only the 'True' methods will be considered in the simulation
METHODS_ALL = {
     'False',  'Max(90% UCB,90% LCB)'; 
     'True',  'Uniformly random';
     'True', 'random on the relevelant features';
     'True', 'max variance';
     'False', 'Bayes experiment design';
     'False',  'Expected information gain';
     'False', 'Bayes experiment design (tr.ref)';
     'False',  'Expected information gain (post_pred)'
     'True', 'Expected information gain (post_pred), fast approx'
     };
Method_list = [];
for m = 1:size(METHODS_ALL,1)
    if strcmp(METHODS_ALL(m,1),'True')
        Method_list = [Method_list,METHODS_ALL(m,2)];
    end
end
num_methods = size(Method_list,2); %number of decision making methods that we want to consider
%% Main algorithm
Loss_1 = zeros(num_methods, num_iterations, num_runs);
Loss_2 = zeros(num_methods, num_iterations, num_runs);
Loss_3 = zeros(num_methods, num_iterations, num_runs);
Loss_4 = zeros(num_methods, num_iterations, num_runs);

decisions = zeros(num_methods, num_iterations, num_runs); 
tic
for run = 1:num_runs 
    disp(['run number ', num2str(run), ' from ', num2str(num_runs), '. acc time = ', num2str(toc) ]);
    %% select the appropriate dataset (simulation or real data)
    if strcmp(DATA_SOURCE,'SIMULATION_DATA') %% Use Synthetic Data
        %Theta_star is the true value of the unknown weight vector
        % non-zero elements of theta_star are generated based on the model parameters
        theta_star = model_params.Nu_theta*randn( num_nonzero_features, 1); 
        theta_star = [theta_star; zeros(num_features-num_nonzero_features,1)]; % make it sparse
        z_star = theta_star ~= 0; % the true value for the latent variable Z in spike and slab model
        %generate new data for each run (because the results is sensitive to the covariate values)
        X_all   = generate_data(num_data,num_features, normalization_method);
        X_train = X_all(1:num_trainingdata,:)'; % select a subset of data as training data
        X_test  = X_all(num_trainingdata+1:num_data,:)'; % the rest are the test data
        Y_train = normrnd(X_train'*theta_star, model_params.Nu_y); % calculate drug responses of the training data
        %Tomi suggested that it makes more sense to use Y_test instead of X_test'*theta_star in the loss functions
        Y_test  = normrnd(X_test'*theta_star, model_params.Nu_y); % calculate drug responses of the test data
    end
    if strcmp(DATA_SOURCE,'GDSC_DATA') %% Use GDSC data set
        %TODO: THIS IS NOT WORKING! CHECK TEMP SCRIPT
        drug_number = 1;
        [X_all, Y_all, theta_star] = load_GDSC_data(drug_number);
        Y_based_on_ground_truth = X_all*theta_star;
        prediction_MSE_ground_truth = sum((Y_all-Y_based_on_ground_truth).^2);
        %TODO: the error is too high for the true theta!! I have to check it with Ammad
        %TODO: I have to divide the test and the training data since the sample size should be quite small
    end
    for method_num = 1:num_methods
        method_name = Method_list(method_num);
        %Feedback = values (1st column) and indices (2nd column) of user feedback
        Feedback = [];
        sparse_options.si = []; % carry prior site terms between interactions
        for it = 1:num_iterations %number of user feedback
            posterior = calculate_posterior(X_train, Y_train, Feedback, model_params, MODE, sparse_params, sparse_options);
            sparse_options.si = posterior.si;
            %% calculate different loss functions
            Loss_1(method_num, it, run) = mean((X_test'*posterior.mean- Y_test).^2);
            Loss_2(method_num, it, run) = mean((posterior.mean-theta_star).^2);       
            %log of posterior predictive dist as the loss function
            %for test data
            post_pred_var = diag(X_test'*posterior.sigma*X_test) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_test'*posterior.mean - Y_test).^2)./(2*post_pred_var);
            Loss_3(method_num, it, run) =  mean(log_post_pred);
            %for training data
            post_pred_var = diag(X_train'*posterior.sigma*X_train) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_train'*posterior.mean - Y_train).^2)./(2*post_pred_var);
            Loss_4(method_num, it, run) = mean(log_post_pred);
            %% make decisions based on a decision policy
            feature_index = decision_policy(posterior, method_name, num_nonzero_features, X_train, Y_train, Feedback, model_params, MODE, sparse_params, sparse_options);
            decisions(method_num, it, run) = feature_index;
            %simulate user feedback
            new_fb_value = user_feedback(feature_index, theta_star, z_star, MODE, model_params);
            Feedback = [Feedback; new_fb_value , feature_index];
        end
    end
end
% profile off
%% averaging and plotting
save('results', 'Loss_1', 'Loss_2', 'Loss_3', 'Loss_4', 'decisions', 'model_params', ...
    'num_nonzero_features', 'Method_list',  'num_features','num_trainingdata', 'MODE', 'normalization_method')
evaluate_results
