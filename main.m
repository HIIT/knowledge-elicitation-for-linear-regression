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
num_methods = 5; %number of decision making methods that we want to consider
num_runs = 50; 
%% Simulator setup
%assume a linear finction
theta_star = 0.5*randn( num_nonzero_features, 1); % We are using randn to generate thethat start
% theta_star = [0.2;0.7;1]; 
theta_star = [theta_star; zeros(num_features-num_nonzero_features,1)];

%% it seems that the results are dependent to our assumptions on X_all 
% %normalization method 1: spike paper
% n=20;
% X_all = rand(n,num_features);% this is all the data
% X_all = X_all./repmat(sqrt(var(X_all)), n,1); %var of each feature should be one
% X_all = X_all - repmat(mean(X_all), n,1);% data should be zero mean
% %normalization method 1.5: spike paper, alternative
% X_all = mvnrnd(zeros(num_features,1), 0.4*eye(num_features,num_features),300);

% normalization method 2: unit vectors
X_all = rand(300,num_features);% this is all the data
X_all = X_all./repmat(sqrt(sum(X_all.^2,2)),1,num_features); %normalize X_all into a unit vector
X = [X_all(1:num_trainingdata,:)]'; % a subset of data that with available drug response
Y = normrnd(X'*theta_star, model_parameters.Nu_y);
% load ('XY')
% save('XY','X','Y');
%% Main algorithm
Loss_1 = zeros(num_methods, num_iterations, num_runs);
Loss_2 = zeros(num_methods, num_iterations, num_runs);
decisions = zeros(num_methods, num_iterations, num_runs); 
for method = 1:num_methods
    method
    for run = 1:num_runs
        Theta_user = []; %user feedback which is a (N_user * 2) array containing [feedback value, feature_number].
        for it = 1:num_iterations
            posterior = calculate_posterior(X, Y, Theta_user, model_parameters);
            Posterior_mean = posterior.mean;
            Loss_1(method, it, run) = sum((X_all*Posterior_mean- X_all*theta_star).^2);
            Loss_2(method, it, run) = sum((Posterior_mean-theta_star).^2);       
            %make decisions based on a decision policy
            feature_index = decision_policy(posterior, method, num_nonzero_features, X, Y, Theta_user, model_parameters);
            decisions(method, it, run) = feature_index;
            %simulate user feedback 
            Theta_user = [Theta_user; normrnd(theta_star(feature_index),model_parameters.Nu_user), feature_index];
        end
    end
end

%% averaging and plotting
save('results', 'Loss_1', 'Loss_2','decisions')
evaluate_results
