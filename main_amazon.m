close all
clear all

%% Parameters and Simulator setup
MODE = 2; 

%The following line loads X_all, Y_all, and keywords (name of features)
load('DATA_amazon\amazon_data');
%The following line loads sparse parameters learned by CV and also theta_star 
% and P_gamma leaned by using all the data.
load('DATA_amazon\cv_results');

%data parameters
num_features       = size(X_all,2);
num_data           = size(X_all,1);
num_trainingdata   = 30;

% define z_star in a meaningful way (either by ground truth or expert opinion)
decision_threshold = 0.9; %this should be on [0.5,1). 
z_star = zeros(num_features,1);
z_star(P_gamma>=decision_threshold) = 1;  %relevant features
z_star(P_gamma<=1-decision_threshold) = 0; %non-relevant features 
z_star(P_gamma<decision_threshold & P_gamma>1-decision_threshold) = -1; %"don't know" features 

%simulation parameters
num_iterations   = 200; %total number of user feedback
num_runs         = 100;

%things that have not been used (since we do not simulate the data)
normalization_method  = -1; % (NOT USED HERE)

%model parameters based on CV results
model_params   = struct('Nu_y',sqrt(sparse_params.sigma2), 'Nu_theta', sqrt(sparse_params.tau2), 'P_user', decision_threshold, 'P_zero', sparse_params.rho);
sparse_options = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0, 'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
% sparse_params  = struct('sigma2',1, 'tau2', 0.1^2 ,'p_u', model_params.P_user,'rho', 0.3 );
sparse_params.p_u = model_params.P_user;
sparse_params.eta2 = -1;   % (NOT USED IN MODE=2)   
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
    %% divide data into training and test
    %randomly select the training data
    train_indices  = false(num_data,1);
    selected_train = datasample(1:num_data,num_trainingdata,'Replace',false);
    train_indices(selected_train) = true;
    test_indices  = ~train_indices;
    
    X_test        = X_all(test_indices,:)'; 
    Y_test        = Y_all(test_indices);   
    X_train       = X_all(train_indices,:)';
    Y_train       = Y_all(train_indices,:);
    %% normalize the data 
    y_mean  = mean(Y_train);
    y_std   = std(Y_train);  
    Y_train = (Y_train - y_mean)./y_std;
    x_mean  = mean(X_train,2);
    x_std   = std(X_train')'; 
    %some of the x_stds can be zero if training size is small. don't divide the data by std if std==0
    x_std(x_std==0) = 1;
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
            % transform predictions back to the original scale
            yhat = X_test'*posterior.mean;
            yhat = yhat .* y_std + y_mean; 
            Loss_1(method_num, it, run) = mean((yhat - Y_test).^2);
            Loss_2(method_num, it, run) = mean((posterior.mean-theta_star).^2);     
            %log of posterior predictive dist as the loss function for test data
            post_pred_var = diag(X_test'*posterior.sigma*X_test) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((yhat - Y_test).^2)./(2*post_pred_var);
            Loss_3(method_num, it, run) =  mean(log_post_pred);
            %log of posterior predictive dist as the loss function for training data
            post_pred_var = diag(X_train'*posterior.sigma*X_train) + model_params.Nu_y^2;
            log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_train'*posterior.mean - Y_train).^2)./(2*post_pred_var);
            Loss_4(method_num, it, run) = mean(log_post_pred);
            %% make decisions based on a decision policy
            feature_index = decision_policy(posterior, method_name, z_star, X_train, Y_train, Feedback, model_params, MODE, sparse_params, sparse_options);
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
    'z_star', 'Method_list',  'num_features','num_trainingdata', 'MODE', 'normalization_method')
evaluate_results
