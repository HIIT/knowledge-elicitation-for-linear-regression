%This scripts is used to learn the best parameters and also to find the best z_star
clear all
close all

load('amazon_data.mat');
%% Learn z_star based on all the data.
num_features = size(X_all,2);
num_data     = size(X_all,1);
num_test     = 200;
num_train    = num_data - num_test;

num_runs     = 10;

%model parameters
sparse_params  = struct('sigma2',1^2, 'tau2', 0.1^2, 'rho', 0.3);
sparse_options = struct('damp',0.5, 'damp_decay',1, 'robust_updates',2, 'verbosity',0, 'max_iter',100, 'threshold',1e-5, 'min_site_prec',1e-6);


methods = {'Ridge', 'Spike and slab'};  
num_methods = 2; 
MSE = zeros(num_methods, num_runs);
tic
for run = 1:num_runs 
    disp([' run number ', num2str(run), ' from ', num2str(num_runs), '. acc time = ', num2str(toc) ]);
    %% divide data into training and test
    %randomly select the training data
    train_indices  = false(num_data,1);
    selected_train = datasample(1:num_data,num_train,'Replace',false);
    train_indices(selected_train) = true;
    test_indices  = ~train_indices;
    X_test        = X_all(test_indices,:);
    Y_test        = Y_all(test_indices);
    X_train       = X_all(train_indices,:);
    Y_train       = Y_all(train_indices,:);
    
    %% normalize the data based on the explanations in the paper
    y_mean = mean(Y_train);
    y_std  = std(Y_train);
    Y_train = (Y_train - y_mean)./y_std;
    Y_test = (Y_test - y_mean)./y_std;
    x_mean  = mean(X_train,1);
    x_std   = std(X_train);
    X_train = bsxfun(@minus,X_train,x_mean);
    X_train = bsxfun(@rdivide, X_train, x_std);
    X_test  = bsxfun(@minus,X_test,x_mean);
    X_test = bsxfun(@rdivide, X_test, x_std);   
    
    %% learn the model
    %for spike-slab
    [fa, si, converged] = linreg_sns_ep(Y_train, X_train, sparse_params, sparse_options);
    if converged ~= 1
        disp(['linreg_sns_ep did not converge for run ', num2str(run)])
    end
    mean_ss    = fa.w.Mean;   
    P_gamma    = fa.P_gamma;
    %for ridge regression
    mean_ridge = ((1 / sparse_params.sigma2) * (X_train' * X_train) + (1 / sparse_params.tau2) * eye(num_features)) \ ((1 / sparse_params.sigma2) * (X_train' * Y_train));
    
    %% calculate error for ridge and spike-slab regression
    MSE(1,run) = mean((X_test*mean_ss- Y_test).^2);
    MSE(2,run) = mean((X_test*mean_ridge- Y_test).^2);
end

average_MSE = mean(MSE,2);
disp([' average MSE for spike and slab after ', num2str(num_runs), ' runs = ', num2str(average_MSE(1))]);
disp([' average MSE for Ridge regression after ', num2str(num_runs), ' runs = ', num2str(average_MSE(2))]);


%% use all the data to find the optimal weight values
% normalize the data first
y_mean = mean(Y_all);
y_std  = std(Y_all);
Y_all_new = (Y_all - y_mean)./y_std;
x_mean  = mean(X_all,1);
x_std   = std(X_all);
X_all_new = bsxfun(@minus,X_all,x_mean);
X_all_new = bsxfun(@rdivide, X_all_new, x_std);


%for spike-slab
[fa, si, converged] = linreg_sns_ep(Y_all_new, X_all_new, sparse_params, sparse_options);
if converged ~= 1
    disp(['linreg_sns_ep did not converge for the current set of params ']);
end
mean_ss    = fa.w.Mean;
P_gamma    = fa.P_gamma;
%for ridge regression
mean_ridge = ((1 / sparse_params.sigma2) * (X_all_new' * X_all_new) + (1 / sparse_params.tau2) * eye(num_features)) \ ((1 / sparse_params.sigma2) * (X_all_new' * Y_all_new));

figure
subplot(1, 2, 1);
plot(mean_ss,'or');
title('Spike and slab model');
xlabel('keywords');
ylabel('mean of posterior weights')
subplot(1, 2, 2);
plot(mean_ridge,'ob');
title('Ridge regression model');
xlabel('keywords');
ylabel('mean of posterior weights')

figure
plot(P_gamma, 'rx');
xlabel('keywords');
ylabel('posterior inclusion probability')

figure
hold on
plot(mean_ss,'.r');
xlabel('keywords');
ylabel('mean of posterior weights')
for i=1:num_features
   if mean_ss(i) > 0.026 || mean_ss(i) < -0.026
       text(i,mean_ss(i),keywords(i),'HorizontalAlignment','right')
   end
end

% save('z_star','z_star');