%This scripts is used to learn the best parameters and also to find the best z_star
clear all
close all

load('amazon_data.mat');
%% Learn z_star based on all the data.
num_features = size(X_all,2);
num_data     = size(X_all,1);
num_train    = 500;
num_test     = num_data - num_train;

num_runs     = 10;

%model parameters
sparse_options = struct('damp',0.8, 'damp_decay',0.95, 'robust_updates',2, 'verbosity',0, 'max_iter',1000, 'threshold',1e-5, 'min_site_prec',1e-6);
sparse_params_s = {struct('sigma2',1^2, 'tau2', 0.1^2, 'rho', 0.1),...
                   struct('sigma2',1^2, 'tau2', 0.5^2, 'rho', 0.1),...
                   struct('sigma2',1^2, 'tau2', 1^2, 'rho', 0.1),...
                   struct('sigma2',1^2, 'tau2', 0.1^2, 'rho', 0.3),...
                   struct('sigma2',1^2, 'tau2', 0.5^2, 'rho', 0.3),...
                   struct('sigma2',1^2, 'tau2', 1^2, 'rho', 0.3),...
                   struct('sigma2',1^2, 'tau2', 0.1^2, 'rho', 0.5),...
                   struct('sigma2',1^2, 'tau2', 0.5^2, 'rho', 0.5),...
                   struct('sigma2',1^2, 'tau2', 1^2, 'rho', 0.5),...
                   struct('sigma2',1^2, 'tau2', 0.1^2, 'rho', 0.8),...
                   struct('sigma2',1^2, 'tau2', 0.5^2, 'rho', 0.8),...
                   struct('sigma2',1^2, 'tau2', 1^2, 'rho', 0.8),...
                   struct('sigma2',0.5^2, 'tau2', 0.1^2, 'rho', 0.1),...
                   struct('sigma2',0.5^2, 'tau2', 0.5^2, 'rho', 0.1),...
                   struct('sigma2',0.5^2, 'tau2', 1^2, 'rho', 0.1),...
                   struct('sigma2',0.5^2, 'tau2', 0.1^2, 'rho', 0.3),...
                   struct('sigma2',0.5^2, 'tau2', 0.5^2, 'rho', 0.3),...
                   struct('sigma2',0.5^2, 'tau2', 1^2, 'rho', 0.3),...
                   struct('sigma2',0.5^2, 'tau2', 0.1^2, 'rho', 0.5),...
                   struct('sigma2',0.5^2, 'tau2', 0.5^2, 'rho', 0.5),...
                   struct('sigma2',0.5^2, 'tau2', 1^2, 'rho', 0.5),...
                   struct('sigma2',0.5^2, 'tau2', 0.1^2, 'rho', 0.8),...
                   struct('sigma2',0.5^2, 'tau2', 0.5^2, 'rho', 0.8),...
                   struct('sigma2',0.5^2, 'tau2', 1^2, 'rho', 0.8),...
                   };
num_params = length(sparse_params_s);

methods = {'Ridge', 'Spike and slab'};  
num_methods = 2; 
MSE = zeros(num_methods, num_runs, num_params);
tic
for iparams = 1:num_params
    disp([' params number ', num2str(iparams), ' from ', num2str(num_params), '. acc time = ', num2str(toc) ]);
    
    sparse_params  = sparse_params_s{iparams};
    
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
        %Y_test = (Y_test - y_mean)./y_std; % predictions are transformed back to the original scale
        x_mean  = mean(X_train,1);
        x_std   = std(X_train); 
        %some of the x_stds can be zero if training size is small. don't divide the data by std if std==0
        x_std(x_std==0) = 1;
        X_train = bsxfun(@minus,X_train,x_mean);
        X_train = bsxfun(@rdivide, X_train, x_std);
        X_test  = bsxfun(@minus,X_test,x_mean);
        X_test = bsxfun(@rdivide, X_test, x_std);

        % remove variables with no variation
        inds = x_std > 0;
        X_train = X_train(:, inds);
        X_test = X_test(:, inds);

        %% learn the model
        %for spike-slab
        [fa, si, converged] = linreg_sns_ep(Y_train, X_train, sparse_params, sparse_options);
        if converged ~= 1
            disp(['linreg_sns_ep did not converge for run ', num2str(run)])
        end
        mean_ss    = fa.w.Mean;
        P_gamma    = fa.P_gamma;
        %for ridge regression
        mean_ridge = ((1 / sparse_params.sigma2) * (X_train' * X_train) + (1 / sparse_params.tau2) * eye(size(X_train,2))) \ ((1 / sparse_params.sigma2) * (X_train' * Y_train));

        %% calculate error for ridge and spike-slab regression (use original scale)
        yhat_ss = X_test*mean_ss;
        yhat_ss = yhat_ss * y_std + y_mean;
        yhat_ridge = X_test*mean_ridge;
        yhat_ridge = yhat_ridge * y_std + y_mean;

        MSE(1,run,iparams) = mean((yhat_ss - Y_test).^2);
        MSE(2,run,iparams) = mean((yhat_ridge - Y_test).^2);
    end
end

average_MSE = min(mean(MSE,2), [], 3);
disp(['best average MSE for spike and slab after ', num2str(num_runs), ' runs = ', num2str(average_MSE(1))]);
disp(['best average MSE for Ridge regression after ', num2str(num_runs), ' runs = ', num2str(average_MSE(2))]);


%% use all the data to find the optimal weight values
% normalize the data first
y_mean = mean(Y_all);
y_std  = std(Y_all);
Y_all_new = (Y_all - y_mean)./y_std;
x_mean  = mean(X_all,1);
x_std   = std(X_all);
%some of the x_stds can be zero if training size is small. don't divide the data by std if std==0
x_std(x_std==0) = 1;
X_all_new = bsxfun(@minus,X_all,x_mean);
X_all_new = bsxfun(@rdivide, X_all_new, x_std);

% Use the best settings from above; note, however, that the best settings
% for small training dataset are not necessarily the best settings for large
% training dataset so this might not be even near optimal for the full data
% if training set was small above.
[~, min_i] = min(mean(MSE(1,:,:),2), [], 3);
sparse_params = sparse_params_s{min_i};

%for spike-slab
[fa, si, converged] = linreg_sns_ep(Y_all_new, X_all_new, sparse_params, sparse_options);
if converged ~= 1
    disp('linreg_sns_ep did not converge for the current set of params ');
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

%% Save all the important parameters based on cross-validation and the solution on all data
theta_star = mean_ss;
save('cv_results','P_gamma','theta_star','sparse_params');