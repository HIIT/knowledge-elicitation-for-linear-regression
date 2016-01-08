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
num_iterations = 15;
Theta_user = []; %user feedback which is a (N_user * 2) array containing [feedback value, feature_number].
%% Simulator setup
%assume a linear finction
theta_star = [0.2;0.7;1];
theta_star = [theta_star; zeros(num_features-size(theta_star,1),1)];
X = 4*rand(num_features,num_traingdata); %% TO DO : I have to add two sets of Xs
Y = normrnd(X'*theta_star, model_parameters.Nu_y);
% load ('XY')
save('XY','X','Y');
%% Main algorithm
Loss = zeros (1,num_iterations);
Posterior_means = zeros(num_iterations,num_features);
for it = 1:num_iterations
    posterior_samples = sample_posterior(X, Y, Theta_user, model_parameters);
    Posterior_means(it,:) = mean(posterior_samples);
    Posterior_means(it,:)
%     Loss(it) = sum((X'*(posterior_mean'-theta_star)).^2);
    Loss(it) = sum((Posterior_means(it,:)'-theta_star).^2)
    %make decisions based on a decision policy
    feature_index = decision_policy(posterior_samples);
    %simulate user feedback 
    Theta_user = [Theta_user; normrnd(theta_star(feature_index),model_parameters.Nu_user), feature_index];
    
end

save('posterior_means','Posterior_means')
%% Plotting
plot(Loss,'.')