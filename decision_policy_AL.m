function [ new_selected_data ] = decision_policy_AL(posterior, Method_name, X_train, Y_train, X_user, ...
    selected_data, sparse_params, sparse_options)
%ACTIVE LEARNING DECISION_POLICY chooses one of the data to be added to training set. 
% Inputs:
% X             covariates (d x n)
% Y             response values
%   _user: unseen data point, _train: training data
% selected_data data that have been selected by active learning to be added from _user data to _training data

    num_features   = size(X_train,1);
    num_user_data  = size(X_user,2); 

    if isfield(sparse_params, 'sigma2_prior') && sparse_params.sigma2_prior
        residual_var = 1 / posterior.fa.sigma2.imean;
    else
        residual_var = sparse_params.sigma2;
    end
    
    
    if strcmp(Method_name,'AL:Uniformly random') 
        %randomly choose one the data points from _user data
        new_selected_data = ceil(rand*num_user_data);        
        if size(selected_data,1)~= 0           
            %Ask about every data only once
            remains = setdiff(1:num_user_data,selected_data);
            if size(remains,2) ~= 0
                new_selected_data = remains(ceil(rand*size(remains,2)));               
            end
        end
    end
    
        
    %TODO: double check the mathematical derivation from Seeger paper.
    if strcmp(Method_name,'AL: Expected information gain')
        %information gain is the KL-divergence of the posterior after and
        %before the user feedback. the expectation is taken over the
        %posterior predictive of new data point. The derivations are based
        %on the paper "Bayesian Inference and Optimal Design for the Sparse
        %Linear Model".

        sigmax   = posterior.sigma * X_user;
        xTsigmax = sum(X_user .* sigmax, 1)';
                
        alpha = 1 + xTsigmax ./ residual_var;
        Utility = log(alpha) + (1 ./ alpha - 1) + ...
            (alpha - 1) ./ (alpha.^2 * residual_var^2) .* (xTsigmax + residual_var);
        Utility(selected_data) = -inf;
           
        [~,new_selected_data]= max(Utility);
        
    end
    
end
    
    
