close all
clear all

%% Parameters
%data parameters
num_features = 30; % total number of features
num_nonzero_features = 5; % features that are nonzero --- AKA sparsity measure
num_trainingdata = 10; % number of samples (patients with available drug response)
% One could measure to check the method is to fix the following ration: #num_traingdata/num_features

%model parameters
model_parameters = struct('Nu_y',0.5, 'Nu_theta', 1, 'Nu_user', 0.1);

%Algorithm parameters
num_iterations = 100;
num_runs = 50;
num_data = 1000 + num_trainingdata; % total number of data (training and test) - this is not important

%% METHOD LIST
% Set the desirable methods to 'True' and others to 'False'
% only the 'True' methods will be considered in the simulation
METHODS_ALL = {
     'True', 'Max(90% UCB,90% LCB)'; 
     'Flase', 'Uniformly random';
     'False', 'random on the relevelant features';
     'True', 'max variance';
     'False', 'Bayes experiment design';
     'False', 'Bayes experiment design (tr.ref)';
     };
Method_list = [];
for m = 1:size(METHODS_ALL,1)
    if strcmp(METHODS_ALL(m,1),'True')
        Method_list = [Method_list,METHODS_ALL(m,2)];
    end
end
num_methods = size(Method_list,2); %number of decision making methods that we want to consider
%% Simulator setup 
normalization_method = 1; %normalization method for generating the data

%% Main algorithm
Loss_1 = zeros(num_methods, num_iterations, num_runs);
Loss_2 = zeros(num_methods, num_iterations, num_runs);
Loss_3 = zeros(num_methods, num_iterations, num_runs);
Loss_4 = zeros(num_methods, num_iterations, num_runs);

decisions = zeros(num_methods, num_iterations, num_runs); 
for run = 1:num_runs 
    run
    %Theta_star is the true value of the unknown weight vector
    theta_star = 0.5*randn( num_nonzero_features, 1); % We are using randn to generate theta start
    theta_star = [theta_star; zeros(num_features-num_nonzero_features,1)]; % make it sparse

    %generate new data for each run (because the results is sensitive to the covariate values)
    X_all   = generate_data(num_data,num_features, normalization_method);
    X_train = X_all(1:num_trainingdata,:)'; % select a subset of data as training data
    X_test  = X_all(num_trainingdata+1:num_data,:)'; % the rest are the test data
    Y_train = normrnd(X_train'*theta_star, model_parameters.Nu_y); % calculate drug responses of the training data
    %Tomi suggested that it makes more sense to use Y_test instead of X_test'*theta_star in the loss functions
    Y_test  = normrnd(X_test'*theta_star, model_parameters.Nu_y); % calculate drug responses of the test data
    for method_num = 1:num_methods
        method_name = Method_list(method_num);
        Theta_user = []; %user feedback which is a (N_user * 2) array containing [feedback value, feature_number].
        for it = 1:num_iterations
            posterior = calculate_posterior(X_train, Y_train, Theta_user, model_parameters);
            Posterior_mean = posterior.mean;
            %% calculate different loss functions
            Loss_1(method_num, it, run) = sum((X_test'*Posterior_mean- Y_test).^2);
            Loss_2(method_num, it, run) = sum((Posterior_mean-theta_star).^2);       
            %log of posterior predictive dist as the loss function           
            for i=1: size(X_test,2)
                post_pred_var = X_test(:,i)'*posterior.sigma*X_test(:,i) + model_parameters.Nu_y^2;
                log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_test(:,i)'*Posterior_mean - Y_test(i))^2)/(2*post_pred_var);    
                Loss_3(method_num, it, run) = Loss_3(method_num, it, run) + log_post_pred;
            end
            for i=1: size(X_train,2)
                post_pred_var = X_train(:,i)'*posterior.sigma*X_train(:,i) + model_parameters.Nu_y^2;
                log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_train(:,i)'*Posterior_mean - Y_train(i))^2)/(2*post_pred_var);
                Loss_4(method_num, it, run) = Loss_4(method_num, it, run) + log_post_pred;
            end
            %% make decisions based on a decision policy
            feature_index = decision_policy(posterior, method_name, num_nonzero_features, X_train, Y_train, Theta_user, model_parameters);
            decisions(method_num, it, run) = feature_index;
            %simulate user feedback 
            Theta_user = [Theta_user; normrnd(theta_star(feature_index),model_parameters.Nu_user), feature_index];
        end
    end
end

%% averaging and plotting
save('results', 'Loss_1', 'Loss_2', 'Loss_3', 'Loss_4', 'decisions', 'num_nonzero_features', 'Method_list')
evaluate_results
