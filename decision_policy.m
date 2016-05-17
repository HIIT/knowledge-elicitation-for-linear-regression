function [ selected_feature ] = decision_policy( posterior , Method_name, num_nonzero_features, X, Y, Theta_user, model_parameters )
%DECISION_POLICY chooses one of the features to show to the user. 

    num_features = size(posterior.mean,1);
    
    if strcmp(Method_name,'Max(90% UCB,90% LCB)') %Combination of UCB and LCB
        % Assume that the features of Theta are independent
        % use 0.9 percentile for now       
        UBs = abs(posterior.mean) + 1.28155 * sqrt(diag(posterior.sigma));
        [~,selected_feature] = max(UBs);
    end
    
    if strcmp(Method_name,'Uniformly random') %randomly choose one feature
        selected_feature = ceil(rand*num_features);
    end
    
    if strcmp(Method_name,'random on the relevelant features') %randomly choose one of the nonzero features
        selected_feature = ceil(rand*num_nonzero_features);
    end
    
    if strcmp(Method_name,'max variance')%choose the feature with highest posterior variance
        %Assume that features of Theta are independent
        VARs = diag(posterior.sigma);
        [~,selected_feature]= max(VARs);
    end
             
    if strcmp(Method_name,'Bayes experiment design') %Bayesian experimental design (with prediction as the goal and expected gain in Shannon information as the utility)         
        Utility = zeros(num_features,1);
        num_data = size(X,2);
        for j=1: num_features
            %Calculate the posterior variance assuming feature j has been selected
            %add a dummy fedback value of 1 to jth feature
            new_theta_user = [Theta_user; 1 , j];
            new_posterior = calculate_posterior(X, Y, new_theta_user, model_parameters);
            for i=1:num_data
                Utility(j) = Utility(j) -0.5*( 1+ log(2*pi*( X(:,i)'*new_posterior.sigma*X(:,i) + model_parameters.Nu_y^2   ) ) );
            end        
        end
        [~,selected_feature]= max(Utility);          
    end

    %TODO: double check this function (it was implemented by Tomi)
    if strcmp(Method_name,'Bayes experiment design (tr.ref)') %Bayesian experimental design (training data reference) %TODO: At the moment this method only selects only one feature (ask Tomi)
        Utility = zeros(num_features,1);
        num_data = size(X,2);
        for j=1:num_features
            %Calculate the posterior variance assuming feature j has been selected
            %add the posterior mean feedback value to jth feature (note:
            %this has no effect on the variance)
            % (a bit lazily coded, but hopefully correct)
            new_theta_user = [Theta_user; posterior.mean(j), j];
            new_posterior = calculate_posterior(X, Y, new_theta_user, model_parameters);
            old_mu = posterior.sigma \ posterior.mean;
            new_f = 0 * old_mu;
            new_f(j) = posterior.mean(j) / model_parameters.Nu_user^2;
            new_f_v = 0 * posterior.sigma;
            new_f_v(j, j) = posterior.sigma(j, j) + model_parameters.Nu_user^2 + posterior.mean(j)^2;
            new_f_v(j, j) = new_f_v(j, j) / model_parameters.Nu_user^4;
            mmt = new_posterior.sigma * (old_mu * old_mu' ...
                     + old_mu * new_f' + new_f * old_mu' ...
                     + new_f_v) * new_posterior.sigma;
            for i=1:num_data
                new_var = X(:,i)' * new_posterior.sigma * X(:,i) + model_parameters.Nu_y^2;
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
        %Linear Model". Unfortunately, the decision policy again only
        %depends on the covariance of the posterior! (and not the mean)
        
        Utility = zeros(num_features,1);
        for j=1: num_features
            %create the feature vector of user feedback
            s = zeros(num_features, 1 ); 
            s(j) = 1;
            alpha = 1 + model_parameters.Nu_user^(-2) * s'*posterior.sigma*s;
            Utility(j) = log(alpha) + (1/alpha -1) + (alpha-1)/(alpha^2 * model_parameters.Nu_user^4) * (s'*posterior.sigma*s + model_parameters.Nu_user^2 );
        end
        [~,selected_feature]= max(Utility);          
    
    end
    
end

