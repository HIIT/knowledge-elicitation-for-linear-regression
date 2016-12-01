close all
clear all
RNG_SEED = rng;

%% Parameters and Simulator setup
MODE = 2; 
% MODE specifies the  type of feedback and the model that we are using 
%           1: Feedback on weight values. Model: spike and slab prior
%           2: Feedback on relevance of features. Model: spike and slab prior

%data parameters for simulation data
num_features         = 100; % total number of features
num_trainingdata     = 10;   % number of training samples
num_userdata         = 500; %data that will be used in active learning
num_data             = 500 + num_trainingdata + num_userdata; % total number of data (training and test)
num_nonzero_features = 10;  % features that are nonzero

%Algorithm parameters
num_iterations = 100 + 1;  %total number of user feedback
num_runs       = 500;  %total number of runs (necessary for averaging results)

%model parameters
sparse_params  = struct('sigma2',1^2, 'tau2', 1^2 , 'eta2',0.1^2,'p_u', 0.95, ...
    'rho', num_nonzero_features/num_features , 'simulated_data', 1);
sparse_options = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0, ...
    'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
%% METHOD LIST
% Set the desirable methods to 'True' and others to 'False'. only the 'True' methods will be considered in the simulation
METHODS_ED = {     
     'True',  'Random';
     'True',  'First relevant features, then non-relevant';
     'False',  'Max posterior inclusion probability';
     'False', 'max variance';
     'False',  'Expected information gain, full EP approx';
     'False',  'Expected information gain, full EP approx, non-sequential';
     'True',  'Expected information gain, fast approx'; %fast approx methdos are available for MODE = 2 only
     'True',  'Expected information gain, fast approx, non-sequential' %fast approx methdos are available for MODE = 2 only
     };
METHODS_AL = {
     'False',  'AL:Uniformly random';
     'False',  'AL: Expected information gain'
     }; 
METHODS_GT = {
     'False',  'Ground truth - all data';
     'True',  'Ground truth - all feedback'
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
Loss_1 = zeros(num_methods, num_iterations, num_runs); % MSE on test
Loss_2 = zeros(num_methods, num_iterations, num_runs); % MSE on train
decisions = zeros(num_methods, num_iterations, num_runs); 
tic
for run = 1:num_runs 
    disp(['run number ', num2str(run), ' from ', num2str(num_runs), '. acc time = ', num2str(toc) ]);
    %% create the simulated data
    %Theta_star is the true value of the unknown weight vector
    % non-zero elements of theta_star are generated based on the model parameters
    theta_star = sqrt(sparse_params.tau2)*randn( num_nonzero_features, 1);
    theta_star = [theta_star; zeros(num_features-num_nonzero_features,1)]; % make it sparse
    z_star = theta_star ~= 0; % the true value for the latent variable Z in spike and slab model
    %generate new data for each run (because the results is sensitive to the covariate values)
    X_all   = mvnrnd(zeros(num_features,1), 1.0*eye(num_features,num_features),num_data);
    Y_all   = normrnd(X_all*theta_star, sqrt(sparse_params.sigma2));
    [X_train, X_user, X_test, Y_train, Y_user, Y_test] = partition_data(X_all, Y_all, num_userdata, num_trainingdata);
    %In the simulation, Generate user feedbacks beforehand so that all methods would receive the same feedback value.
    all_feedback = zeros(num_features,1);
    for feature_index = 1:size(X_train,1)
        new_fb_value = user_feedback(feature_index, theta_star, z_star, MODE, sparse_params);
        all_feedback(feature_index) = new_fb_value;
    end
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
                posterior = calculate_posterior([X_train, X_user], [Y_train; Y_user], [], ...
                     MODE, sparse_params, sparse_options);
            end
            if find(strcmp('Ground truth - all feedback', method_name))
                %calculate the posterior based on all feedbacks
                for feature_index = 1:size(X_train,1)
                    %new_fb_value = user_feedback(feature_index, theta_star, z_star, MODE, sparse_params);
                    Feedback = [Feedback; all_feedback(feature_index) , feature_index];
                end
                 posterior = calculate_posterior(X_train, Y_train, Feedback, ...
                     MODE, sparse_params, sparse_options);                                            
            end
            Y_hat = X_test'*posterior.mean;
            Y_hat_train = X_train'*posterior.mean;
            Loss_1(method_num, :, run) = mean((Y_hat- Y_test).^2); %MSE
            Loss_2(method_num, :, run) = mean((Y_hat_train- Y_train).^2); %MSE on training  
            continue
        end
        %% for non-sequential ED methods find the suggested queries before user interaction
        if strfind(char(method_name),'non-sequential')
            posterior = calculate_posterior(X_train, Y_train, [], MODE, sparse_params, sparse_options);
            %find non-sequential order of features to be queried from the user
            non_seq_feature_indices = decision_policy(posterior, method_name, z_star, X_train, Y_train, ...
                [], MODE, sparse_params, sparse_options);
        end
        %% User interaction
        for it = 1:num_iterations %number of user feedback
            %calculate the posterior based on training + feedback until now
            posterior = calculate_posterior([X_train, X_user(:,selected_data)], [Y_train; Y_user(selected_data)], Feedback, ...
                MODE, sparse_params, sparse_options);
            sparse_options.si = posterior.si;
            %% calculate different loss functions
            Y_hat = X_test'*posterior.mean;
            Y_hat_train = X_train'*posterior.mean;
            Loss_1(method_num, it, run) = mean((Y_hat- Y_test).^2); %MSE
            Loss_2(method_num, it, run) = mean((Y_hat_train- Y_train).^2); %MSE on training  
            %% If ED: make a decision based on ED decision policy
            if find(strcmp(Method_list_ED, method_name))               
                if strfind(char(method_name),'non-sequential')
                    %for non-sequential methods, use the saved order
                    if it<=num_features                   
                        feature_index = non_seq_feature_indices(it);
                    else
                        %suggest random feature if all feedbacks are already given 
                        feature_index = ceil(rand*num_features);
                    end
                else
                    %for sequential methods find the next decision based on feedback until now
                    feature_index = decision_policy(posterior, method_name, z_star, X_train, Y_train, ...
                        Feedback, MODE, sparse_params, sparse_options);
                end
                decisions(method_num, it, run) = feature_index;
                %simulate user feedback (next line), or just read it from the saved array
                %new_fb_value = user_feedback(feature_index, theta_star, z_star, MODE, sparse_params);
                Feedback = [Feedback; all_feedback(feature_index) , feature_index];
            end
            %% If AL: add a new data point based on AL decision policy 
            if find(strcmp(Method_list_AL, method_name))               
                [new_selected_data] = decision_policy_AL(posterior, method_name, ...
                    [X_train, X_user(:,selected_data)] , [Y_train; Y_user(selected_data)], ...
                    X_user, selected_data, sparse_params, sparse_options);                
                selected_data = [selected_data;new_selected_data]; 
            end
        end
    end
end
%% averaging and plotting
save('results', 'Loss_1', 'Loss_2', 'decisions', 'sparse_options', 'sparse_params', ...
    'z_star', 'Method_list',  'num_features','num_trainingdata', 'MODE', 'RNG_SEED')
evaluate_results
