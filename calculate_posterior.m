function [ posterior ] = calculate_posterior(X, Y, feedback, model_params, MODE, sparse_params, sparse_options)
% Calculate the posterior distribution given the observed data and user feedback
% Inputs:
% MODE          Feedback type: 0 and 1: noisy observation of weight. 2: binary relevance of feature
%               0: analytical solution for the posterior with multivariate Gaussian prior
%               1: Multivariate Gaussian approximation of the posterior with spike and slab prior
%               2: Joint posterior approximation with EP.
% X             covariates (d x n)
% Y             response values
% feedback      values (1st column) and indices (2nd column) of feedback (n_feedbacks x 2)

    if MODE == 1
        %assume sparse prior (spike and slab) and approximate the posterior
        %with a multivariate Gaussian distribution using EP algorithm
        [fa, si, converged] = linreg_sns_ep(Y, X', sparse_params, sparse_options, feedback, [], sparse_options.si);
        posterior.si = si;
        posterior.sigma = inv(fa.w.Tau);
        posterior.mean  = fa.w.Mean;        
        
    end
    if MODE == 0
        %calculate the analytical solution for posterior with Gaussian prior
        num_features = size(X,1);
        num_userfeedback = size(feedback,1);
        posterior = struct('mean',zeros(num_features,1), 'sigma', zeros(num_features,num_features));

        sigma_inverse = (1/model_params.Nu_theta)^2 * eye(num_features); % part 1

        X = X'; %design matrix
        sigma_inverse = sigma_inverse + (1/model_params.Nu_y)^2 * X'*X; %part 2 

        temp = 0;
        %if user has given feedback
        if num_userfeedback > 0
            F = feedback(:,1); % user feedbacks
            S = zeros(num_userfeedback, num_features); %design matrix for user feedback
            for i=1:num_userfeedback
                S(i,feedback(i,2)) = 1;
            end
            sigma_inverse = sigma_inverse + (1/model_params.Nu_user)^2 * S'*S; %part 3
            temp = (1/model_params.Nu_user)^2 * S'*F;
        end

        posterior.sigma = inv(sigma_inverse);
        posterior.mean  = posterior.sigma * ( (1/model_params.Nu_y)^2 * X'*Y + temp );
    end
    %TODO: this posterior.si is required for the sparse case. fix the function interfaces later and remove this line
    posterior.si = [];
    
    
    if MODE == 2        
        %assume sparse prior (spike and slab) and approximate the posterior with EP
        %weights are approximated by a multivariate Gaussian distribution.
        %latent variables are approximated by Bernoulli distribution. 
        [fa, si, converged] = linreg_sns_ep(Y, X', sparse_params, sparse_options, [] , feedback, sparse_options.si);
        posterior.si = si;
        posterior.sigma = inv(fa.w.Tau);
        posterior.mean  = fa.w.Mean;       
        posterior.p   = fa.gamma.p; % TODO: VERIFY THIS LATER
        
        
    end
end