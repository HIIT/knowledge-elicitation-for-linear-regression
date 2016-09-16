function [ new_selected_data ] = decision_policy_AL(posterior, Method_name, X_train, Y_train, X_user, ...
    selected_data, model_params, sparse_params, sparse_options)
%ACTIVE LEARNING DECISION_POLICY chooses one of the data to be added to training set. 
% Inputs:
% X             covariates (d x n)
% Y             response values
%   _user: unseen data point, _train: training data
% selected_data data that have been selected by active learning to be added from _user data to _training data

    num_features   = size(X_train,1);
    num_user_data  = size(X_user,2); 
    
    %randomly choose one the data points from _user data
    if strcmp(Method_name,'AL:Uniformly random') 
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
        
        
        alpha = 1 + xTsigmax ./ model_params.Nu_y^2;
        Utility = log(alpha) + (1 ./ alpha - 1) + ...
            (alpha - 1) ./ (alpha.^2 * model_params.Nu_y^4) .* (xTsigmax + model_params.Nu_y^2);
        Utility(selected_data) = -inf;
        
%         %This is the older (and slower) version of the above code.
%         %I did not delete it because it also works for the case where "s" is not a basis vector.
%         Utility = zeros(num_user_data,1);
%         for j=1: num_user_data
%             %if the data has been selected before, ignore it
%             if ismember(j,selected_data)
%                 Utility(j) = -inf;
%                 continue
%             end
%             x = X_user(:,j);
%             xTsigmax = x'*posterior.sigma*x;
%             alpha = 1 + model_params.Nu_y^(-2) * xTsigmax;
%             Utility(j) = log(alpha) + (1/alpha -1) + ...
%                 (alpha-1)/(alpha^2 * model_params.Nu_y^4) * (xTsigmax + model_params.Nu_y^2 );
%         end       

        [~,new_selected_data]= max(Utility);
        
    end
    
end
    
    
