function [ selected_feature ] = decision_policy( posterior , Method_name, z_star, X, Y, Feedback, MODE, sparse_params, sparse_options)
%DECISION_POLICY chooses one of the features to query to the user. 
% Inputs:
% MODE          Feedback type: 1: noisy observation of weight. 2: binary relevance of feature
%               1: Multivariate Gaussian approximation of the posterior with spike and slab prior
%               2: Joint posterior approximation with EP.
% X             covariates (d x n)
% Y             response values
% feedback      values (1st column) and indices (2nd column) of feedback (n_feedbacks x 2)

    num_features = size(posterior.mean,1);
    
    if isfield(sparse_params, 'sigma2_prior') && sparse_params.sigma2_prior
        residual_var = 1 / posterior.fa.sigma2.imean;
    else
        residual_var = sparse_params.sigma2;
    end

    
    if strcmp(Method_name,'Random') 
        %Randomly choose one feature
        selected_feature = ceil(rand*num_features);      
        if size(Feedback,1)~= 0  
            %ask about each feature only once
            remains = setdiff(1:num_features,Feedback(:,2));
            if size(remains,2) ~= 0
                selected_feature = remains(ceil(rand*size(remains,2)));               
            end
        end
    end
        
    
    if strcmp(Method_name,'First relevant features, then non-relevant') 
        %randomly choose one of the relevant features at first and then start asking about random features (oracle decision maker)
        relevants = find(z_star == 1)';        
        selected_feature = relevants(ceil(rand*size(relevants,2)));
        if size(Feedback,1)~= 0
            %ask about each relevant feature only once
            remains_relevant = setdiff(relevants,Feedback(:,2));
            if size(remains_relevant,2) ~= 0
                selected_feature = remains_relevant(ceil(rand*size(remains_relevant,2)));
            else 
                %after asking about all relevants, ask about random features
                remains = setdiff(1:num_features,Feedback(:,2)); 
                if size(remains,2) ~= 0
                    selected_feature = remains(ceil(rand*size(remains,2)));
                end
            end
        end
    end
    
      
    if strcmp(Method_name,'max variance')
        %Choose the feature with highest posterior variance
        %Assume that features of Theta are independent
        Utility = diag(posterior.sigma);
        % ask about each feature only once
        if ~isempty(Feedback)
            Utility(Feedback(:,2)) = -inf;
        end
        [~,selected_feature] = max(Utility);
    end
    
    
    if strfind(char(Method_name),'Max posterior inclusion probability')
        % Choose the feature that has the largest posterior inclusion
        % probability (and has not been given feedback yet).        
        Utility = posterior.p;   
        % ask about each feature only once 
        if ~isempty(Feedback)
            Utility(Feedback(:,2)) = -inf;
        end       
        [~,selected_feature] = max(Utility);
    end
    
    
    %% If the feedback is on the value of coefficients 
    if  MODE == 1
 
        if  strfind(char(Method_name),'Expected information gain, full EP approx') 
            %information gain is the KL-divergence of the posterior_predictive 
            %of the data after and before the user feedback. 
            %the expectation is taken over the posterior predictive of user feedback. 
            %The derivations are based on the AISTATS submission [TODO: add a reference here]. 
            %This code works for sequential and non-sequential case

            alpha = 1 + diag(posterior.sigma) / sparse_params.eta2;

            sTsigmas = diag(posterior.sigma);
            sigmax = posterior.sigma * X;
            xTsigmax = sum(X .* sigmax, 1);
            xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha * sparse_params.eta2));

            part1 = 0.5 * log(bsxfun(@rdivide, xTsigmax + residual_var, xTsigma_newx + residual_var));
            part2_numerator = xTsigma_newx + residual_var + bsxfun(@times, bsxfun(@rdivide, sigmax, alpha * residual_var).^2, sTsigmas + residual_var);
            part2_denumerator = 2 * (xTsigmax + residual_var);

            Utility = sum(part1 + bsxfun(@rdivide, part2_numerator , part2_denumerator) - 0.5, 2);

    %         %This is the older (and slower) version of the above code. 
    %         %I did not delete it because it also works for the case where "s" is not a basis vector.
    %         Utility = zeros(num_features,1);         
    %         for j=1: num_features           
    %             %create the feature vector of user feedback
    %             s = zeros(num_features, 1 );
    %             s(j) = 1;
    %             %calculate alpha [notes]
    %             alpha = 1 + 1/sparse_params.eta2 * s'*posterior.sigma*s;
    %             %calculate the new covarianse matrix considering s
    %             sigma_new = posterior.sigma - 1/sparse_params.eta2 * 1/alpha * (posterior.sigma * s) * (s' * posterior.sigma);            
    %             %some temp variable that make the calculations cleaner
    %             sTsigmas = s'*posterior.sigma*s;
    %             xTsigmax = diag(X'*posterior.sigma*X);
    %             xTsigma_newx = diag(X'*sigma_new*X);
    %             xTsigmas = X'*(posterior.sigma*s);
    %            
    %             %expected information gain formula: 
    %             part1 = 0.5 * log( (xTsigmax + residual_var)./(xTsigma_newx + residual_var) );              
    %             part2_numerator = xTsigma_newx + residual_var + ...
    %                 (1/residual_var *1/alpha*xTsigmas).^2 * (sTsigmas + residual_var);
    %             part2_denumerator = 2*(xTsigmax + residual_var);
    %                             
    %             Utility(j) = sum(part1 + part2_numerator./part2_denumerator - 0.5);                          
    %         end
    
            % ask about each feature only once 
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end    
            
            if strfind(char(Method_name),'non-sequential')
                %for non-sequential case
                %sort the utility function and send all indices
                [~,selected_feature] = sort(Utility,'descend');
            else
                %for sequential case
                [~,selected_feature] = max(Utility);
            end  
        end
        
    end
    
    %% If the feedback is on the relevance of coefficients
    if MODE == 2
        
        if strfind(char(Method_name),'Expected information gain, full EP approx') 
            %information gain is the KL-divergence of the posterior_predictive 
            %of the data after and before the user feedback. 
            %the expectation is taken over the posterior predictive of user feedback.  
            %The derivations are based on the AISTATS submission [TODO: add a reference here]. 
            %This code works for sequential and non-sequential case   
            
            Utility = zeros(num_features,1);
            
            %some temp variables
            sigmax = posterior.sigma * X;
            xTsigmax = sum(X .* sigmax, 1);
            xMu = X' * posterior.mean;
            part2_denumerator = 2 * (xTsigmax + residual_var);
            for j=1: num_features
                %Calculate the KL divergence between posterior predictive after and before feedback
                % if feedback is 1
                %add a fedback value for the jth feature and calculate the new posterior
                new_fb_1 = [Feedback; 1 , j];
                new_posterior_1 = calculate_posterior(X, Y, new_fb_1, MODE, sparse_params, sparse_options);
                sx_1 = new_posterior_1.sigma * X;
                xTsigma_1x = sum(X .* sx_1, 1);
                part1 = 0.5 * log( (xTsigmax + residual_var)./(xTsigma_1x + residual_var) );
                xMu_1 = X' * new_posterior_1.mean; 
                part2_numerator = xTsigma_1x + residual_var + (xMu_1' - xMu').^2;
                
                KL_1 = sum(part1 + part2_numerator ./ part2_denumerator - 0.5, 2);
                
                % if feedback is 0
                %add a fedback value for the jth feature and calculate the new posterior
                new_fb_0 = [Feedback; 0 , j];
                new_posterior_0 = calculate_posterior(X, Y, new_fb_0, MODE, sparse_params, sparse_options);
                sx_0 = new_posterior_0.sigma * X;
                xTsigma_0x = sum(X .* sx_0, 1);
                part1 = 0.5 * log( (xTsigmax + residual_var)./(xTsigma_0x + residual_var) );
                xMu_0 = X' * new_posterior_0.mean;
                part2_numerator = xTsigma_0x + residual_var + (xMu_0' - xMu').^2;
                KL_0 = sum(part1 + part2_numerator ./ part2_denumerator - 0.5, 2);
                
                %Calculate the E[KL], where expectation is on the posterior predictive of the feedback value
                post_pred_f0 =   sparse_params.p_u + posterior.p(j) - 2*sparse_params.p_u * posterior.p(j);
                
                Utility(j) = post_pred_f0 * KL_0 + (1-post_pred_f0) * KL_1;
            end
            
            % ask about each feature only once
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            
            if strfind(char(Method_name),'non-sequential')
                %for non-sequential case
                %sort the utility function and send all indices
                [~,selected_feature] = sort(Utility,'descend');
            else
                %for sequential case
                [~,selected_feature] = max(Utility);
            end  
              
        end
        
        
        if strfind(char(Method_name),'Expected information gain, fast approx')
            % information gain is the KL-divergence of the posterior_predictive
            % of the data after and before the user feedback.
            % the expectation is taken over the posterior predictive of user feedback.
            % We approximate the posterior given candidate feedback by running
            % one step EP update for the relevance feedback followed by one step EP
            % update for the corresponding prior site.
            % The derivations are based on the AISTATS submission [TODO: add a reference here]. 
            % This code works for sequential and non-sequential case  
            
            % TODO: compute updates only for candidates that have not been given feedback?
            
            pr = sparse_params;
            op = sparse_options;
            op.damp = 1; % don't damp updates?
            
            % feedback predictive distribution:
            post_pred_f0 = sparse_params.p_u + posterior.p - 2 * sparse_params.p_u * posterior.p;
            
            % KLs
            KL_0 = compute_post_pred_kl(0, posterior, pr, op, X, sparse_params);
            KL_1 = compute_post_pred_kl(1, posterior, pr, op, X, sparse_params);
      
            Utility = post_pred_f0 .* KL_0 + (1 - post_pred_f0) .* KL_1;
            
            % ask about each feature only once 
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            
            if strfind(char(Method_name),'non-sequential')
                %for non-sequential case
                %sort the utility function and send all indices
                [~,selected_feature] = sort(Utility,'descend');
            else
                %for sequential case
                [~,selected_feature] = max(Utility);
            end
            
        end
        
        if strfind(char(Method_name),'LPD, fast approx')
            pr = sparse_params;
            op = sparse_options;
            op.damp = 1; % don't damp updates?
            
            % feedback predictive distribution:
            post_pred_f0 = sparse_params.p_u + posterior.p - 2 * sparse_params.p_u * posterior.p;
            
            % LPDs
            lpd_0 = compute_post_pred_ld(0, posterior, pr, op, X, sparse_params);
            lpd_1 = compute_post_pred_ld(1, posterior, pr, op, X, sparse_params);
      
            Utility = post_pred_f0 .* lpd_0 + (1 - post_pred_f0) .* lpd_1;
            
            % ask about each feature only once
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            
            [~,selected_feature] = max(Utility);
        end

        if strfind(char(Method_name),'Thompson f, fast approx')
            % This version samples from the predictive distribution of
            % feedbacks (one sample) and uses that to weight the LPDs and
            % chooses the feedback that gives the highest LPD.
            
            pr = sparse_params;
            op = sparse_options;
            op.damp = 1; % don't damp updates?
            
            % feedback predictive distribution:
            post_pred_f0 = sparse_params.p_u + posterior.p - 2 * sparse_params.p_u * posterior.p;
            f0 = rand(length(post_pred_f0), 1) < post_pred_f0;
            
            % LPDs
            lpd_0 = compute_post_pred_ld(0, posterior, pr, op, X, sparse_params);
            lpd_1 = compute_post_pred_ld(1, posterior, pr, op, X, sparse_params);
      
            Utility = f0 .* lpd_0 + (1 - f0) .* lpd_1;
            
            % ask about each feature only once
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            
            [~,selected_feature] = max(Utility);
        end
        
        if strfind(char(Method_name),'Thompson f2, fast approx')
            % This version samples from the predictive distribution of
            % feedbacks (multiple samples) and estimates the probabilities
            % that each feature would give the highest LPD and then samples
            % the actual feedback from there.
            
            pr = sparse_params;
            op = sparse_options;
            op.damp = 1; % don't damp updates?
            
            % feedback predictive distribution:
            post_pred_f0 = sparse_params.p_u + posterior.p - 2 * sparse_params.p_u * posterior.p;
            f0 = bsxfun(@lt, rand(length(post_pred_f0), 10000), post_pred_f0);
            
            % LPDs
            lpd_0 = compute_post_pred_ld(0, posterior, pr, op, X, sparse_params);
            lpd_1 = compute_post_pred_ld(1, posterior, pr, op, X, sparse_params);
            
            % ask about each feature only once
            if ~isempty(Feedback)
                lpd_0(Feedback(:,2)) = -inf;
                lpd_1(Feedback(:,2)) = -inf;
            end
            
            Utility = bsxfun(@times, f0, lpd_0) + bsxfun(@times, 1 - f0, lpd_1);
            Utility = sum(bsxfun(@eq, Utility, max(Utility, [], 1)), 2);

            selected_feature = datasample(1:length(Utility), 1, 'Weights', Utility);
        end
        
        if strfind(char(Method_name),'Thompson, fast approx')
            % This version samples from the predictive distribution of y in
            % the current state (without including the new feedback) which
            % might not make much sense.
            % TODO: Now lpd_0 and lpd_1 use different samples! Would make
            % more sense to use the same! (But still not might make too
            % much sense.)
            
            pr = sparse_params;
            op = sparse_options;
            op.damp = 1; % don't damp updates?
            
            % feedback predictive distribution:
            post_pred_f0 = sparse_params.p_u + posterior.p - 2 * sparse_params.p_u * posterior.p;
            
            % Thompson LPDs
            lpd_0 = thompson_lpd(0, posterior, pr, op, X, sparse_params);
            lpd_1 = thompson_lpd(1, posterior, pr, op, X, sparse_params);
      
            Utility = post_pred_f0 .* lpd_0 + (1 - post_pred_f0) .* lpd_1;
            
            % ask about each feature only once
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            
            [~,selected_feature] = max(Utility);
        end
        
    end
end



function kl = compute_post_pred_kl(feedback, posterior, pr, op, X, sparse_params)

sf = posterior.ep_subfunctions;

m = length(posterior.p);
fa = posterior.fa;
si = posterior.si;

pr.m = m;
pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);

% EP updates
ca_gf = sf.compute_bernoulli_lik_cavity(fa.gamma.p_nat, si.gamma_feedback);
ti_gf = sf.compute_bernoulli_lik_tilt(ca_gf, pr, feedback * ones(m, 1));
si.gamma_feedback = sf.update_bernoulli_lik_sites(si.gamma_feedback, ca_gf, ti_gf, op);
fa = sf.compute_full_approximation_gamma(fa, si, pr);
ca_prior = sf.compute_sns_prior_cavity(fa, si.w_prior, pr);
ti_prior = sf.compute_sns_prior_tilt(ca_prior, pr);
si.w_prior = sf.update_sns_prior_sites(si.w_prior, ca_prior, ti_prior, op);

% changes in parameters
delta_tau = si.w_prior.normal_tau - posterior.si.w_prior.normal_tau;
delta_mu = si.w_prior.normal_mu - posterior.si.w_prior.normal_mu;

% KL
alpha = 1 + diag(posterior.sigma) .* delta_tau;

sigmax = posterior.sigma * X;
xTsigmax = sum(X .* sigmax, 1);
xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha ./ delta_tau));

if isfield(pr, 'sigma2_prior') && pr.sigma2_prior
    residual_var = 1 / posterior.fa.sigma2.imean;
else
    residual_var = sparse_params.sigma2;
end

part1 = 0.5 * log(bsxfun(@rdivide, xTsigmax + residual_var, xTsigma_newx + residual_var));
%part2_numerator = xTsigma_newx + model_params.Nu_y^2 + bsxfun(@times, bsxfun(@rdivide, sigmax, alpha * model_params.Nu_y^2).^2, sTsigmas + model_params.Nu_y^2);
part2_numerator = xTsigma_newx + residual_var + bsxfun(@times, sigmax, (posterior.mean .* delta_tau - delta_mu) ./ alpha).^2;
part2_denumerator = 2 * (xTsigmax + residual_var);

kl = sum(part1 + bsxfun(@rdivide, part2_numerator , part2_denumerator) - 0.5, 2);

end

function lpd = compute_post_pred_ld(feedback, posterior, pr, op, X, sparse_params)

sf = posterior.ep_subfunctions;

m = length(posterior.p);
fa = posterior.fa;
si = posterior.si;

pr.m = m;
pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);

% EP updates
ca_gf = sf.compute_bernoulli_lik_cavity(fa.gamma.p_nat, si.gamma_feedback);
ti_gf = sf.compute_bernoulli_lik_tilt(ca_gf, pr, feedback * ones(m, 1));
si.gamma_feedback = sf.update_bernoulli_lik_sites(si.gamma_feedback, ca_gf, ti_gf, op);
fa = sf.compute_full_approximation_gamma(fa, si, pr);
ca_prior = sf.compute_sns_prior_cavity(fa, si.w_prior, pr);
ti_prior = sf.compute_sns_prior_tilt(ca_prior, pr);
si.w_prior = sf.update_sns_prior_sites(si.w_prior, ca_prior, ti_prior, op);

% changes in parameters
delta_tau = si.w_prior.normal_tau - posterior.si.w_prior.normal_tau;
%delta_mu = si.w_prior.normal_mu - posterior.si.w_prior.normal_mu;

if isfield(pr, 'sigma2_prior') && pr.sigma2_prior
    residual_var = 1 / posterior.fa.sigma2.imean;
else
    residual_var = sparse_params.sigma2;
end

% LPD
sigmax = posterior.sigma * X;
xTsigmax = sum(X .* sigmax, 1);
alpha = 1 + diag(posterior.sigma) .* delta_tau;
xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha ./ delta_tau));

lpd = -0.5 * sum(log(xTsigma_newx + residual_var), 2);

end

function lpd = thompson_lpd(feedback, posterior, pr, op, X, model_params)

% slightly adapted Thompson sampling approach
% sample ys from old posterior

if isfield(pr, 'sigma2_prior') && pr.sigma2_prior
    residual_var = 1 / posterior.fa.sigma2.imean;
else
    residual_var = model_params.Nu_y^2;
end

sigmax = posterior.sigma * X;
xTsigmax = sum(X .* sigmax, 1);

y = X' * posterior.mean + sqrt(xTsigmax' + residual_var) .* randn(size(X, 2), 1);


sf = posterior.ep_subfunctions;

m = length(posterior.p);
fa = posterior.fa;
si = posterior.si;

pr.m = m;
pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);

% EP updates
ca_gf = sf.compute_bernoulli_lik_cavity(fa.gamma.p_nat, si.gamma_feedback);
ti_gf = sf.compute_bernoulli_lik_tilt(ca_gf, pr, feedback * ones(m, 1));
si.gamma_feedback = sf.update_bernoulli_lik_sites(si.gamma_feedback, ca_gf, ti_gf, op);
fa = sf.compute_full_approximation_gamma(fa, si, pr);
ca_prior = sf.compute_sns_prior_cavity(fa, si.w_prior, pr);
ti_prior = sf.compute_sns_prior_tilt(ca_prior, pr);
si.w_prior = sf.update_sns_prior_sites(si.w_prior, ca_prior, ti_prior, op);

% changes in parameters
delta_tau = si.w_prior.normal_tau - posterior.si.w_prior.normal_tau;
delta_mu = si.w_prior.normal_mu - posterior.si.w_prior.normal_mu;

% LPD
% -0.5 * log(var) + -0.5 * (y - mean)^2 / var
alpha = 1 + diag(posterior.sigma) .* delta_tau;
xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha ./ delta_tau));
vars = xTsigma_newx + residual_var;
means = bsxfun(@minus, posterior.mean' * X, bsxfun(@times, (posterior.mean .* delta_tau - delta_mu) ./ alpha, sigmax)); 

lpd = -0.5 * sum(log(vars) + bsxfun(@minus, y', means).^2 ./ vars, 2);

end