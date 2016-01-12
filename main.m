close all
clear all

%% At the moment I have convergence problems in MCMC. the problem is that as the %number of observations increases, I need to tune the parameters again, but I cant!
%% I should go with the basic analytical form solution

%% Parameters
%data parameters
num_features = 10;
num_traingdata = 3;

%model parameters
model_parameters = struct('Nu_y',1, 'Nu_theta', 3, 'Nu_user', 0.05);

%Algorithm parameters
num_iterations = 10;
num_methods = 4; %number of decision making methods that we want to consider
num_runs = 10; 
%% Simulator setup
%assume a linear finction
theta_star = [0.2;0.7;1];
theta_star = [theta_star; zeros(num_features-size(theta_star,1),1)];
X = 4*rand(num_features,num_traingdata);
X_unknown = 4*rand(num_features,100); % These are Xs with unknown drug responses
Y = normrnd(X'*theta_star, model_parameters.Nu_y);
% load ('XY')
save('XY','X','Y');
%% Main algorithm
Loss_1 = zeros(num_methods, num_iterations, num_runs);
Loss_2 = zeros(num_methods, num_iterations, num_runs);
decisions = zeros(num_methods, num_iterations, num_runs); 
for method = 1:num_methods
    method
    for run = 1:num_runs
        Theta_user = []; %user feedback which is a (N_user * 2) array containing [feedback value, feature_number].
        Posterior_mean = zeros(1,num_features);
        for it = 1:num_iterations
            posterior_samples = sample_posterior(X, Y, Theta_user, model_parameters);
            Posterior_mean = mean(posterior_samples);
            Loss_1(method, it, run) = sum(([X,X_unknown]'*(Posterior_mean'-theta_star)).^2);
            Loss_2(method, it, run) = sum((Posterior_mean'-theta_star).^2);       
            %make decisions based on a decision policy
            feature_index = decision_policy(posterior_samples, method);
            decisions(method, it, run) = feature_index;
            %simulate user feedback 
            Theta_user = [Theta_user; normrnd(theta_star(feature_index),model_parameters.Nu_user), feature_index];
        end
    end
end

%% averaging and plotting

save('Loss_functions', 'Loss_1', 'Loss_2')
save('decisions','decisions')

evaluate_results
