close all
clear all

%% Parameters
MODE = 2; 
%The following line loads X_all, Y_all, and keywords (name of features)
load('DATA_amazon\amazon_data');
%The following line loads z_star
load('DATA_amazon\z_star');

num_features = size(X_all,2);
num_data     = size(X_all,1);
num_folds    = 5;  % number of partitions
num_trainingdata    = num_data- floor(num_data/num_folds);
%Algorithm parameters
num_iterations   = 10; %total number of user feedback


%things that have not been used (since we do not simulate the data
theta_star = zeros(num_features,1);  % (NOT USED HERE)
num_nonzero_features            = 0; % (NOT USED HERE)
normalization_method            = 0; % (NOT USED HERE)

%model parameters
model_params   = struct('Nu_y',1, 'Nu_theta', 0.1, 'Nu_user', 0.1, 'P_user', 0.9, 'P_zero', 0.3);
sparse_options = struct('damp',0.5, 'damp_decay',1, 'robust_updates',2, 'verbosity',0, 'max_iter',100, 'threshold',1e-5, 'min_site_prec',1e-6);
sparse_params  = struct('sigma2',model_params.Nu_y^2, 'tau2', model_params.Nu_theta^2 ,'eta2',model_params.Nu_user^2,'p_u', model_params.P_user);
sparse_params.rho = model_params.P_zero;
%% METHOD LIST
% Set the desirable methods to 'True' and others to 'False'. only the 'True' methods will be considered in the simulation
METHODS_ALL = {
     'False',  'Max(90% UCB,90% LCB)'; 
     'True',  'Uniformly random';
     'False', 'random on the relevelant features';
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
Loss_1 = zeros(num_methods, num_iterations, num_folds);
Loss_2 = zeros(num_methods, num_iterations, num_folds);  % (NOT USED HERE) we do not have theta_star for real data
Loss_3 = zeros(num_methods, num_iterations, num_folds);
Loss_4 = zeros(num_methods, num_iterations, num_folds);

decisions = zeros(num_methods, num_iterations, num_folds); 
tic

%divide data in k_fold clusters
K_fold_indices = crossvalind('Kfold', num_data, num_folds);

for fold = 1:num_folds 
    disp([' fold number ', num2str(fold), ' from ', num2str(num_folds), '. acc time = ', num2str(toc) ]);
    %% divide data into training and test data in k clusters 
    test_indices = (K_fold_indices == fold);
    X_test        = X_all(test_indices,:)'; 
    Y_test        = Y_all(test_indices);   
    train_indices = ~test_indices;
    X_train       = X_all(train_indices,:)';
    Y_train       = Y_all(train_indices,:);
    %% normalize the data 
    y_mean  = mean(Y_train);
    y_std   = std(Y_train);  
    Y_train = (Y_train - y_mean)./y_std;
    Y_test  = (Y_test - y_mean)./y_std; 
    x_mean  = mean(X_train,2);
    x_std   = std(X_train')';
    X_train = bsxfun(@minus,X_train,x_mean);
    X_train = bsxfun(@rdivide, X_train, x_std);
    X_test  = bsxfun(@minus,X_test,x_mean);
    X_test = bsxfun(@rdivide, X_test, x_std);
    
    for method_num = 1:num_methods
        method_name = Method_list(method_num);
        %Feedback = values (1st column) and indices (2nd column) of user feedback
        Feedback = [];
        sparse_options.si = []; % carry prior site terms between interactions
        for it = 1:num_iterations %number of user feedback
            posterior = calculate_posterior(X_train, Y_train, Feedback, model_params, MODE, sparse_params, sparse_options);
            sparse_options.si = posterior.si;
            %% calculate different loss functions
            Loss_1(method_num, it, fold) = mean((X_test'*posterior.mean- Y_test).^2);
%             Loss_2(method_num, it, fold) = mean((posterior.mean-theta_star).^2); % (NOT USED HERE)     
            %log of posterior predictive dist as the loss function
            %for test data
            post_pred_var = diag(X_test'*posterior.sigma*X_test) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_test'*posterior.mean - Y_test).^2)./(2*post_pred_var);
            Loss_3(method_num, it, fold) =  mean(log_post_pred);
            %for training data
            post_pred_var = diag(X_train'*posterior.sigma*X_train) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_train'*posterior.mean - Y_train).^2)./(2*post_pred_var);
            Loss_4(method_num, it, fold) = mean(log_post_pred);
            %% make decisions based on a decision policy
            feature_index = decision_policy(posterior, method_name, num_nonzero_features, X_train, Y_train, Feedback, model_params, MODE, sparse_params, sparse_options);
            decisions(method_num, it, fold) = feature_index;
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
