function [ selected_feature ] = decision_policy( posterior , Method_name, num_nonzero_features, X, Y, Feedback, model_params, MODE, sparse_params, sparse_options)
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
        
        if MODE == 2 && size(Feedback,1)~= 0           
            %ask about each feature only once
            remains = setdiff(1:num_features,Feedback(:,2));
            if size(remains,2) ~= 0
                selected_feature = remains(ceil(rand*size(remains,2)));               
            end
        end
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
        if MODE == 2 
            % ask about each feature only once (note: one must not ask for more feedbacks than the number of features
            % or this will start giving 1 always)
            if ~isempty(Feedback)
                VARs(Feedback(:,2)) = -inf;
            end
            [~,selected_feature] = max(VARs);
        end
    end
    
    % If the feedback is on the value of features
    if MODE == 0 || MODE == 1
        
 
        if strcmp(Method_name,'Bayes experiment design') 
            %TODO: We do not need to call the calculate posterior function here
            %again. Just use the iterative formula in Expected information gain (post_pred) implementation 
            %Bayesian experimental design (with prediction as the goal and expected gain in Shannon information as the utility)
            Utility = zeros(num_features,1);
            for j=1: num_features
                %Calculate the posterior variance assuming feature j has been selected
                %add a dummy fedback value of 1 to jth feature
                new_theta_user = [Feedback; 1 , j];
                new_posterior = calculate_posterior(X, Y, new_theta_user, model_params, MODE, sparse_params, sparse_options);
                for i=1:num_data
                    Utility(j) = Utility(j) -0.5*( 1+ log(2*pi*( X(:,i)'*new_posterior.sigma*X(:,i) + model_params.Nu_y^2   ) ) );
                end        
            end
            [~,selected_feature]= max(Utility);          
        end


 
        if strcmp(Method_name,'Bayes experiment design (tr.ref)') 
            %TODO: double check this function (it was implemented by Tomi)
            %TODO: At the moment this method only selects only one feature (ask Tomi)
            %Bayesian experimental design (training data reference)
            Utility = zeros(num_features,1);
            for j=1:num_features
                %Calculate the posterior variance assuming feature j has been selected
                %add the posterior mean feedback value to jth feature (note:
                %this has no effect on the variance)
                % (a bit lazily coded, but hopefully correct)
                new_theta_user = [Feedback; posterior.mean(j), j];
                new_posterior = calculate_posterior(X, Y, new_theta_user, model_params, MODE, sparse_params, sparse_options);
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
            %The derivations are based on the notes for Coupling_Bandits. 

            alpha = 1 + diag(posterior.sigma) / model_params.Nu_user^2;

            sTsigmas = diag(posterior.sigma);
            sigmax = posterior.sigma * X;
            xTsigmax = sum(X .* sigmax, 1);
            xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha * model_params.Nu_user^2));

            part1 = 0.5 * log(bsxfun(@rdivide, xTsigmax + model_params.Nu_y^2, xTsigma_newx + model_params.Nu_y^2));
            part2_numerator = xTsigma_newx + model_params.Nu_y^2 + bsxfun(@times, bsxfun(@rdivide, sigmax, alpha * model_params.Nu_y^2).^2, sTsigmas + model_params.Nu_y^2);
            part2_denumerator = 2 * (xTsigmax + model_params.Nu_y^2);

            Utility = sum(part1 + bsxfun(@rdivide, part2_numerator , part2_denumerator) - 0.5, 2);

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
    %             part2_denumerator = 2*(xTsigmax + model_params.Nu_y^2);
    %                             
    %             Utility(j) = sum(part1 + part2_numerator./part2_denumerator - 0.5);                          
    %         end

            [~,selected_feature]= max(Utility);   
        end
        
    end
    
    % If the feedback is on the relevance of features
    if MODE == 2
        
        if strcmp(Method_name,'Expected information gain (post_pred)') 
            %information gain is the KL-divergence of the posterior_predictive 
            %of the data after and before the user feedback. 
            %the expectation is taken over the posterior predictive of user feedback. 
            %The derivations are based on the notes for Coupling_Bandits. 
            
            Utility = zeros(num_features,1);
            
            %some temp variables
            sigmax = posterior.sigma * X;
            xTsigmax = sum(X .* sigmax, 1);
            xMu = X' * posterior.mean;
            part2_denumerator = 2 * (xTsigmax + model_params.Nu_y^2);
            for j=1: num_features
                %Calculate the KL divergence between posterior predictive after and before feedback
                % if feedback is 1
                %add a fedback value for the jth feature and calculate the new posterior
                new_fb_1 = [Feedback; 1 , j];
                new_posterior_1 = calculate_posterior(X, Y, new_fb_1, model_params, MODE, sparse_params, sparse_options);
                sx_1 = new_posterior_1.sigma * X;
                xTsigma_1x = sum(X .* sx_1, 1);
                part1 = 0.5 * log( (xTsigmax + model_params.Nu_y^2)./(xTsigma_1x + model_params.Nu_y^2) );
                xMu_1 = X' * new_posterior_1.mean; 
                part2_numerator = xTsigma_1x + model_params.Nu_y^2 + (xMu_1' - xMu').^2;
                
                KL_1 = sum(part1 + part2_numerator ./ part2_denumerator - 0.5, 2);
                
                % if feedback is 0
                %add a fedback value for the jth feature and calculate the new posterior
                new_fb_0 = [Feedback; 0 , j];
                new_posterior_0 = calculate_posterior(X, Y, new_fb_0, model_params, MODE, sparse_params, sparse_options);
                sx_0 = new_posterior_0.sigma * X;
                xTsigma_0x = sum(X .* sx_0, 1);
                part1 = 0.5 * log( (xTsigmax + model_params.Nu_y^2)./(xTsigma_0x + model_params.Nu_y^2) );
                xMu_0 = X' * new_posterior_0.mean;
                part2_numerator = xTsigma_0x + model_params.Nu_y^2 + (xMu_0' - xMu').^2;
                KL_0 = sum(part1 + part2_numerator ./ part2_denumerator - 0.5, 2);
                
                %Calculate the E[KL], where expectation is on the posterior predictive of the feedback value
                post_pred_f0 =   model_params.P_user + posterior.p(j) - 2*model_params.P_user * posterior.p(j);
                
                Utility(j) = post_pred_f0 * KL_0 + (1-post_pred_f0) * KL_1;
            end
            
            % ask about each feature only once (note: one must not ask for more feedbacks than the number of features
            % or this will start giving 1 always)
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            [~,selected_feature] = max(Utility);
              
        end
        
        if strcmp(Method_name,'Expected information gain (post_pred), fast approx')
            % information gain is the KL-divergence of the posterior_predictive
            % of the data after and before the user feedback.
            % the expectation is taken over the posterior predictive of user feedback.
            % TODO: The derivations are based on the notes for Coupling_Bandits.
            % We approximate the posterior given candidate feedback by running
            % one step EP update for the relevance feedback followed by one step EP
            % update for the corresponding prior site.
            
            % TODO: compute updates only for candidates that have not been
            % given feedback?
            
            pr = sparse_params;
            op = sparse_options;
            op.damp = 1; % don't damp updates?
            
            % feedback predictive distribution:
            post_pred_f0 = model_params.P_user + posterior.p - 2 * model_params.P_user * posterior.p;
            
            % KLs
            KL_0 = compute_post_pred_kl(0, posterior, pr, op, X, model_params);
            KL_1 = compute_post_pred_kl(1, posterior, pr, op, X, model_params);
      
            Utility = post_pred_f0 .* KL_0 + (1 - post_pred_f0) .* KL_1;
            
            % ask about each feature only once (note: one must not ask for more feedbacks than the number of features
            % or this will start giving 1 always)
            if ~isempty(Feedback)
                Utility(Feedback(:,2)) = -inf;
            end
            [~,selected_feature] = max(Utility);
        end
    end
end



function kl = compute_post_pred_kl(feedback, posterior, pr, op, X, model_params)

m = length(posterior.p);
fa = posterior.fa;
si = posterior.si;
feedbacks = [(feedback * ones(m, 1)) (1:m)'];

pr.m = m;
pr.rho_nat = log(pr.rho) - log1p(-pr.rho);
pr.p_u_nat = log(pr.p_u) - log1p(-pr.p_u);

% EP updates
ca_gf = compute_gf_cavity(fa, si.gf);
ti_gf = compute_gf_tilt(ca_gf, pr, feedbacks);
si.gf = update_gf_sites(si.gf, ca_gf, ti_gf, feedbacks, op);
fa = compute_full_approximation_gamma(fa, si, pr);
ca_prior = compute_w_prior_cavity(fa, si.prior, pr);
ti_prior = compute_w_prior_tilt(ca_prior, pr);
si.prior = update_w_prior_sites(si.prior, ca_prior, ti_prior, op);

% changes in parameters
delta_tau = si.prior.w.tau - posterior.si.prior.w.tau;
delta_mu = si.prior.w.mu - posterior.si.prior.w.mu;

% KL
alpha = 1 + diag(posterior.sigma) .* delta_tau;

sigmax = posterior.sigma * X;
xTsigmax = sum(X .* sigmax, 1);
xTsigma_newx = bsxfun(@minus, xTsigmax, bsxfun(@rdivide, sigmax.^2, alpha ./ delta_tau));

part1 = 0.5 * log(bsxfun(@rdivide, xTsigmax + model_params.Nu_y^2, xTsigma_newx + model_params.Nu_y^2));
%part2_numerator = xTsigma_newx + model_params.Nu_y^2 + bsxfun(@times, bsxfun(@rdivide, sigmax, alpha * model_params.Nu_y^2).^2, sTsigmas + model_params.Nu_y^2);
part2_numerator = xTsigma_newx + model_params.Nu_y^2 + bsxfun(@times, sigmax, (posterior.mean .* delta_tau - delta_mu) ./ alpha).^2;
part2_denumerator = 2 * (xTsigmax + model_params.Nu_y^2);

kl = sum(part1 + bsxfun(@rdivide, part2_numerator , part2_denumerator) - 0.5, 2);

end

% TODO: these are copy-pasted from linreg_ss_ep.m. Refactor to use same
% code?

function ca = compute_gf_cavity(fa, si)

ca.gamma.p_nat = fa.gamma.p_nat - si.gamma.p_nat;

end


function ti = compute_gf_tilt(ca, pr, feedbacks)

% feedbacks: first is value, second index.
% Computes only those with feedback:
ti.gamma.mean = 1 ./ (1 + exp(-(ca.gamma.p_nat(feedbacks(:,2)) + (2 * feedbacks(:, 1) - 1) .* pr.p_u_nat)));
ti.gamma.mean = max(min(ti.gamma.mean, 1-eps), eps);

end


function si = update_gf_sites(si, ca, ti, feedbacks, op)

si.gamma.p_nat(feedbacks(:,2)) = (1 - op.damp) * si.gamma.p_nat(feedbacks(:,2)) + op.damp * (log(ti.gamma.mean) - log1p(-ti.gamma.mean) - ca.gamma.p_nat(feedbacks(:,2)));

end


function ca = compute_w_prior_cavity(fa, si, pr)

m = pr.m;

tmp = fa.w.Tau_chol \ eye(m);
var_w = sum(tmp.^2)';

denom = (1 - si.w.tau .* var_w);
ca.w.tau = denom ./ var_w;
ca.w.mean = (fa.w.Mean - var_w .* si.w.mu) ./ denom;

ca.gamma.p_nat = fa.gamma.p_nat - si.gamma.p_nat;
ca.gamma.p = 1 ./ (1 + exp(-ca.gamma.p_nat));

end


function [ti, z] = compute_w_prior_tilt(ca, pr)

t = ca.w.tau + 1 ./ pr.tau2;

g_var = 1 ./ ca.w.tau; % for gamma0
mcav2 = ca.w.mean.^2;
log_z_gamma0 = log1p(-ca.gamma.p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
g_var = pr.tau2 + g_var; % for gamma1
log_z_gamma1 = log(ca.gamma.p) - 0.5 * log(g_var) - 0.5 * mcav2 ./ g_var;
z_gamma0 = exp(log_z_gamma0 - log_z_gamma1);
z_gamma1 = ones(size(log_z_gamma1));
z = 1 + z_gamma0;

ti.w.mean = z_gamma1 .* (ca.w.tau .* ca.w.mean) ./ t ./ z;
e2_w_tilt = z_gamma1 .* (1 ./ t + 1 ./ t.^2 .* (ca.w.tau .* ca.w.mean).^2) ./ z;
ti.w.var = e2_w_tilt - ti.w.mean.^2;

ti.gamma.mean = z_gamma1 ./ z;
ti.gamma.mean = max(min(ti.gamma.mean, 1-eps), eps);

end


function [si, nonpositive_cavity_vars, nonpositive_site_var_proposals] = update_w_prior_sites(si, ca, ti, op)

nonpositive_site_var_proposals = false;

% skip negative cavs
update_inds = ca.w.tau(:) > 0;
nonpositive_cavity_vars = ~all(update_inds);

new_tau_w_site = 1 ./ ti.w.var - ca.w.tau;

switch op.robust_updates
    case 0
    case 1
        inds_tmp = new_tau_w_site(:) > 0;
        nonpositive_site_var_proposals = ~all(inds_tmp);
        update_inds = update_inds & inds_tmp;
    case 2
        inds = new_tau_w_site(:) <= 0;
        new_tau_w_site(inds) = op.min_site_prec;
        ti.w.var(inds) = 1./(op.min_site_prec + ca.w.tau(inds));
end
new_mu_w_site = ti.w.mean ./ ti.w.var - ca.w.tau .* ca.w.mean;
si.w.tau(update_inds) = (1 - op.damp) * si.w.tau(update_inds) + op.damp * new_tau_w_site(update_inds);
si.w.mu(update_inds) = (1 - op.damp) * si.w.mu(update_inds) + op.damp * new_mu_w_site(update_inds);

si.gamma.p_nat(update_inds) = (1 - op.damp) * si.gamma.p_nat(update_inds) + op.damp * (log(ti.gamma.mean(update_inds)) - log1p(-ti.gamma.mean(update_inds)) - ca.gamma.p_nat(update_inds));

end


function fa = compute_full_approximation_gamma(fa, si, pr)

fa.gamma.p_nat = si.prior.gamma.p_nat + si.gf.gamma.p_nat + pr.rho_nat;
fa.gamma.p = 1 ./ (1 + exp(-fa.gamma.p_nat));

end