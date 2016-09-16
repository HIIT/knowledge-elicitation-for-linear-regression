close all
clear all

%% Parameters and Simulator setup
MODE = 2; 
% MODE specifies the  type of feedback and the model that we are using
%           0: Feedback on weight values. Model: Gaussian prior 
%           1: Feedback on weight values. Model: spike and slab prior
%           2: Feedback on relevance of features. Model: spike and slab prior

%data parameters for simulation data
num_features         = 100; % total number of features
num_trainingdata     = 5;   % number of training samples
num_userdata         = 500; %data that will be used in active learning
num_data             = 500 + num_trainingdata + num_userdata; % total number of data (training and test)
num_nonzero_features = 10;  % features that are nonzero

%Algorithm parameters
num_iterations = 100;  %total number of user feedback
num_runs       = 100;  %total number of runs (necessary for averaging results)

%model parameters
normalization_method = 1; %normalization method for generating the data (Xs)
model_params   = struct('Nu_y',0.1, 'Nu_theta', 1, 'Nu_user', 0.1, 'P_user', 0.99, ...
    'P_zero', num_nonzero_features/num_features,  'simulated_data', 1);
sparse_options = struct('damp',0.5, 'damp_decay',1, 'robust_updates',2, 'verbosity',0, ...
    'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
sparse_params  = struct('sigma2',model_params.Nu_y^2, 'tau2', model_params.Nu_theta^2 , ...
    'eta2',model_params.Nu_user^2,'p_u', model_params.P_user);
sparse_params.rho = model_params.P_zero;
%% METHOD LIST
% Set the desirable methods to 'True' and others to 'False'. only the 'True' methods will be considered in the simulation
METHODS_ED = {
     'False',  'Max(90% UCB,90% LCB)'; 
     'True',  'Uniformly random';
     'True', 'random on the relevelant features';
     'True', 'max variance';
     'False', 'Bayes experiment design';
     'False',  'Expected information gain';
     'False', 'Bayes experiment design (tr.ref)';
     'False',  'Expected information gain (post_pred)';
     'True',  'Expected information gain (post_pred), fast approx'
     };
METHODS_AL = {
     'False',  'AL:Uniformly random';
     'False',  'AL: Expected information gain'
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
Method_list = [Method_list_ED,Method_list_AL];
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
    %% create the simulated data
    %Theta_star is the true value of the unknown weight vector
    % non-zero elements of theta_star are generated based on the model parameters
    theta_star = model_params.Nu_theta*randn( num_nonzero_features, 1);
    theta_star = [theta_star; zeros(num_features-num_nonzero_features,1)]; % make it sparse
    z_star = theta_star ~= 0; % the true value for the latent variable Z in spike and slab model
    %generate new data for each run (because the results is sensitive to the covariate values)
    X_all   = generate_data(num_data,num_features, normalization_method);
    Y_all   = normrnd(X_all*theta_star, model_params.Nu_y);
    [X_train, X_user, X_test, Y_train, Y_user, Y_test] = partition_data(X_all, Y_all, num_userdata, num_trainingdata);
    %% main algorithms (ED and AL)
    for method_num = 1:num_methods
        method_name = Method_list(method_num);
        %Feedback = values (1st column) and indices (2nd column) of user feedback
        Feedback = [];            %only used in experimental design methdos
        %selected_data = indices of data selected by active learning from X_user and Y_user
        selected_data = [];       %only used in active learning methods
        sparse_options.si = [];   %carry prior site terms between interactions
        
        %% User interaction
        for it = 1:num_iterations %number of user feedback
            %calculate the posterior based on training + feedback until now
            posterior = calculate_posterior([X_train, X_user(:,selected_data)], [Y_train; Y_user(selected_data)], Feedback, ...
                model_params, MODE, sparse_params, sparse_options);
            sparse_options.si = posterior.si;
            %% calculate different loss functions
            Loss_1(method_num, it, run) = mean((X_test'*posterior.mean- Y_test).^2); %MSE
            Loss_2(method_num, it, run) = mean((posterior.mean-theta_star).^2);       
            %log of posterior predictive dist as the loss function for test data
            post_pred_var = diag(X_test'*posterior.sigma*X_test) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_test'*posterior.mean - Y_test).^2)./(2*post_pred_var);
            Loss_3(method_num, it, run) =  mean(log_post_pred);
            %log of posterior predictive dist as the loss function for training data
            post_pred_var = diag(X_train'*posterior.sigma*X_train) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_train'*posterior.mean - Y_train).^2)./(2*post_pred_var);
            Loss_4(method_num, it, run) = mean(log_post_pred);
            %% If ED: make a decision based on ED decision policy
            if find(strcmp(Method_list_ED, method_name))
                feature_index = decision_policy(posterior, method_name, z_star, X_train, Y_train, ...
                    Feedback, model_params, MODE, sparse_params, sparse_options);
                decisions(method_num, it, run) = feature_index;
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
%% averaging and plotting
save('results', 'Loss_1', 'Loss_2', 'Loss_3', 'Loss_4', 'decisions', 'model_params', ...
    'z_star', 'Method_list',  'num_features','num_trainingdata', 'MODE', 'normalization_method')
evaluate_results
