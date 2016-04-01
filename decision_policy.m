function [ selected_feature ] = decision_policy( posterior , Method, num_nonzero_features, X, Y, Theta_user, model_parameters )
%DECISION_POLICY chooses one of the features to show to the user
    %Method = 1: UCB, 2: random all, 3: random non zero features
    %Method = 4: feature with highest posterior variance
    %Method = 5: Bayesian experimental design (with prediction as the goal and expected gain in Shannon information as the utility)
    num_features = size(posterior.mean,1);
    if Method == 1 %Combination of UCB and LCB
        % Assume that the features of Theta are independent
        % use 0.9 percentile for now       
        UBs = abs(posterior.mean) + 1.28155 * sqrt(diag(posterior.sigma));
        [~,selected_feature] = max(UBs);
    end
    if Method == 2 %randomly choose one feature
        selected_feature = ceil(rand*num_features);
    end
    if Method == 3 %randomly choose one of the nonzero features
        selected_feature = ceil(rand*num_nonzero_features);
    end
    if Method == 4 %choose the feature with highest posterior variance
        %Assume that features of Theta are independent
        VARs = diag(posterior.sigma);
        [~,selected_feature]= max(VARs);
    end
             
    if Method == 5 %Bayesian experimental design        
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

    if Method == 6 %Bayesian experimental design (training data reference)
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
end

