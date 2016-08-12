function [ X_all ] = generate_data( num_data,num_features, normalization_method )
%GENERATE_DATA generates covariates based on some assumptions

    if normalization_method == 1
        % normalization method 1: unit vectors
        X_all = rand(num_data,num_features);% this is all the data
        X_all = X_all./repmat(sqrt(sum(X_all.^2,2)),1,num_features); %normalize X_all into a unit vector
    end
    if normalization_method == 2
        %normalization method 2: spike paper
        X_all = rand(num_data,num_features);% this is all the data
        X_all = X_all./repmat(sqrt(var(X_all)), num_data,1); %var of each feature should be one
        X_all = X_all - repmat(mean(X_all), num_data,1);% data should be zero mean
    end
    if normalization_method == 3
        %normalization method 3: spike paper, alternative
        X_all = mvnrnd(zeros(num_features,1), 1.0*eye(num_features,num_features),num_data);
    end
    
end

% Note; I also changed the generation of covariate matrix X to standard normal variates. 
% The signal-to-noise ratio in linear regression is something like "max_num_nonzero_features * Nu_theta^2 * var_X / Nu_y^2"
% , where var_X is the variance of a covariate (and assuming all covariates have the same variance). 
% With standard normal variates, var_X is about 1 and, say, Nu_theta^2=1, max_num_nonzero_features=10, and Nu_y^2=5, 
% SNR is something like 2:1 (unless I'm mistaken somewhere). 
% Anyway, the previous data generation setting generates covariates with var_X = 0.0025 which leads to quite small signal 
% (SNR of 10*0.0025/0.5^2=1:10) and shows rather small changes in measures like MSE. 
% Of course, this can be an interesting scenario and feel free to change the parameters back, 
% although I think it would be simpler to generate covariates with var_X around 1 (often variables are standardized before 
% applying linear regression unless they are already on some commeasurate scale) and then control SNR with the other parameters 
% (unless there are other reasons to do something else).