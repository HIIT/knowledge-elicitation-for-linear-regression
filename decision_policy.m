function [ selected_feature ] = decision_policy( posterior , Method_name, num_nonzero_features, X, Y, Theta_user, model_params, mode, sparse_params, sparse_options)
%DECISION_POLICY chooses one of the features to show to the user. 
% TODO: if we consider only Gaussian posterior then the inputs can be trimmed a little bit
    num_features = size(posterior.mean,1);
    num_data = size(X,2);
    
    %Combination of UCB and LCB
    if strcmp(Method_name,'Max(90% UCB,90% LCB)') 
        % Assume that the features of Theta are independent
        % use 0.9 percentile for now       
        UBs = abs(posterior.mean) + 1.28155 * sqrt(diag(posterior.sigma));
        [~,selected_feature] = max(UBs);
    end
    
    
    %randomly choose one feature
    if strcmp(Method_name,'Uniformly random') 
        selected_feature = ceil(rand*num_features);
    end
    
    
    %randomly choose one of the nonzero features
    if strcmp(Method_name,'random on the relevelant features') 
        selected_feature = ceil(rand*num_nonzero_features);
    end
    
    
    %choose the feature with highest posterior variance
    if strcmp(Method_name,'max variance')
        %Assume that features of Theta are independent
        VARs = diag(posterior.sigma);
        [~,selected_feature]= max(VARs);
    end
    
    
    %TODO: We do not need to call the calculate posterior function here
    %again. Just use the iterative formula in Expected information gain (post_pred) implementation 
    %Bayesian experimental design (with prediction as the goal and expected gain in Shannon information as the utility) 
    if strcmp(Method_name,'Bayes experiment design') 
        Utility = zeros(num_features,1);
        for j=1: num_features
            %Calculate the posterior variance assuming feature j has been selected
            %add a dummy fedback value of 1 to jth feature
            new_theta_user = [Theta_user; 1 , j];
            new_posterior = calculate_posterior(X, Y, new_theta_user, model_params, mode, sparse_params, sparse_options);
            for i=1:num_data
                Utility(j) = Utility(j) -0.5*( 1+ log(2*pi*( X(:,i)'*new_posterior.sigma*X(:,i) + model_params.Nu_y^2   ) ) );
            end        
        end
        [~,selected_feature]= max(Utility);          
    end

    
    %TODO: double check this function (it was implemented by Tomi)
    %TODO: At the moment this method only selects only one feature (ask Tomi)
    %Bayesian experimental design (training data reference) 
    if strcmp(Method_name,'Bayes experiment design (tr.ref)') 
        Utility = zeros(num_features,1);
        for j=1:num_features
            %Calculate the posterior variance assuming feature j has been selected
            %add the posterior mean feedback value to jth feature (note:
            %this has no effect on the variance)
            % (a bit lazily coded, but hopefully correct)
            new_theta_user = [Theta_user; posterior.mean(j), j];
            new_posterior = calculate_posterior(X, Y, new_theta_user, model_params, mode, sparse_params, sparse_options);
            old_mu = posterior.sigma \ posterior.mean;
            new_f = 0 * old_mu;
            new_f(j) = posterior.mean(j) / model_params.Nu_user^2;
            new_f_v = 0 * posterior.sigma;
            new_f_v(j, j) = posterior.sigma(j, j) + model_params.Nu_user^2 + posterior.mean(j)^2;
            new_f_v(j, j) = new_f_v(j, j) / model_params.Nu_user^4;
            mmt = new_posterior.sigma * (old_mu * old_mu' ...
                     + old_mu * new_f' + new_f * old_mu' ...
                     + new_f_v) * new_posterior.sigma;
            for i=1:num_data
                new_var = X(:,i)' * new_posterior.sigma * X(:,i) + model_params.Nu_y^2;
                Utility(j) = Utility(j) - 0.5 * log(2 * pi) ...
                             - 0.5 * log(new_var) ...
                             - 0.5 * (Y(i)^2 - 2 * Y(i) * X(:, i)' * new_posterior.mean ...
                                      + X(:, i)' * mmt * X(:, i)) / new_var;
            end
        end
        [~,selected_feature]= max(Utility);
    end
    
    
    %TODO: double check the mathematical derivation from Seeger paper.
    if strcmp(Method_name,'Expected information gain')  
        %information gain is the KL-divergence of the posterior after and
        %before the user feedback. the expectation is taken over the
        %posterior predictive of user feedback. The derivations are based
        %on the paper "Bayesian Inference and Optimal Design for the Sparse
        %Linear Model". 
                
        alpha = 1 + diag(posterior.sigma) / model_params.Nu_user^2;
        Utility = log(alpha) + (1 ./ alpha - 1) + ...
            (alpha - 1) ./ (alpha.^2 * model_params.Nu_user^4) .* (diag(posterior.sigma) + model_params.Nu_user^2);

%         %This is the older (and slower) version of the above code. 
%         %I did not delete it because it also works for the case where "s" is not a basis vector.
%         Utility = zeros(num_features,1); 
%         for j=1: num_features
%             %create the feature vector of user feedback
%             s = zeros(num_features, 1 ); 
%             s(j) = 1;
%             alpha = 1 + model_params.Nu_user^(-2) * s'*posterior.sigma*s;
%             Utility(j) = log(alpha) + (1/alpha -1) + ...
%                 (alpha-1)/(alpha^2 * model_params.Nu_user^4) * (s'*posterior.sigma*s + model_params.Nu_user^2 );
%         end
        
        [~,selected_feature]= max(Utility);

    end
    
    
    if strcmp(Method_name,'Expected information gain (post_pred)') 
        %information gain is the KL-divergence of the posterior_predictive 
        %of the data after and before the user feedback. 
        %the expectation is taken over the posterior predictive of user feedback. 
        %The derivations are based on the notes for Couplong bandits. 

        alpha = 1 + diag(posterior.sigma) / model_params.Nu_user^2;
        
        sTsigmas = diag(posterior.sigma);
        sx = posterior.sigma * X;
        xTsigmax = sum(X .* sx, 1);
        xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sx.^2, alpha * model_params.Nu_user^2));
        
        part1 = 0.5 * log(bsxfun(@rdivide, xTsigmax + model_params.Nu_y^2, xTsigma_newx + model_params.Nu_y^2));
        part2_numerator = xTsigma_newx + model_params.Nu_y^2 + bsxfun(@times, bsxfun(@rdivide, sx, alpha * model_params.Nu_y^2).^2, sTsigmas + model_params.Nu_y^2);
        part2_denumerator = 2 * (xTsigma_newx + model_params.Nu_y^2);
        
        Utility = sum(part1 + part2_numerator ./ part2_denumerator - 0.5, 2);

%         %This is the older (and slower) version of the above code. 
%         %I did not delete it because it also works for the case where "s" is not a basis vector.
%         Utility = zeros(num_features,1);         
%         for j=1: num_features           
%             %create the feature vector of user feedback
%             s = zeros(num_features, 1 );
%             s(j) = 1;
%             %calculate alpha [notes]
%             alpha = 1 + model_params.Nu_user^(-2) * s'*posterior.sigma*s;
%             %calculate the new covarianse matrix considering s
%             sigma_new = posterior.sigma - model_params.Nu_user^(-2) * 1/alpha * (posterior.sigma * s) * (s' * posterior.sigma);            
%             %some temp variable that make the calculations cleaner
%             sTsigmas = s'*posterior.sigma*s;
%             xTsigmax = diag(X'*posterior.sigma*X);
%             xTsigma_newx = diag(X'*sigma_new*X);
%             xTsigmas = X'*(posterior.sigma*s);
%            
%             %expected information gain formula: 
%             part1 = 0.5 * log( (xTsigmax + model_params.Nu_y^2)./(xTsigma_newx + model_params.Nu_y^2) );              
%             part2_numerator = xTsigma_newx + model_params.Nu_y^2 + ...
%                 (model_params.Nu_y^(-2)*1/alpha*xTsigmas).^2 * (sTsigmas + model_params.Nu_y^2);
%             part2_denumerator = 2*(xTsigma_newx + model_params.Nu_y^2);
%                             
%             Utility(j) = sum(part1 + part2_numerator./part2_denumerator - 0.5);                          
%         end
        
        [~,selected_feature]= max(Utility);   
        
    end
    
end

