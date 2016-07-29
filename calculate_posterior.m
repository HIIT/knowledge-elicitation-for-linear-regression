function [ posterior ] = calculate_posterior(X, Y, Theta_user, model_params, mode, sparse_params, sparse_options)
% Calculate the posterior distribution given the observed data and user feedback
% Inputs:
% mode          0: analytical solution for the posterior with multivariate Gaussian prior
%               1: Multivariate Gaussian approximation of the posterior with spike and slab prior
% X             covariates (n x m)
% Y             response values
% Theta_user    values (1st column) and indices (2nd column) of feedback (n_feedbacks x 2)

    if mode == 1
        %assume sparse prior (spike and slab) and approximate the posterior
        %with a multivariate Gaussian distribution using EP algorithm
        [fa, si, converged] = linreg_sns_ep(Y, X', sparse_params, sparse_options, Theta_user, sparse_options.si);
        posterior.si = si;
        posterior.sigma = inv(fa.w.Tau);
        posterior.mean  = fa.w.Mean;        
        
    end
    if mode == 0
        %calculate the analytical solution for posterior with Gaussian prior
        num_features = size(X,1);
        num_userfeedback = size(Theta_user,1);
        posterior = struct('mean',zeros(num_features,1), 'sigma', zeros(num_features,num_features));

        sigma_inverse = (1/model_params.Nu_theta)^2 * eye(num_features); % part 1

        X = X'; %design matrix
        sigma_inverse = sigma_inverse + (1/model_params.Nu_y)^2 * X'*X; %part 2 

        temp = 0;
        %if user has given feedback
        if num_userfeedback > 0
            F = Theta_user(:,1); % user feedbacks
            S = zeros(num_userfeedback, num_features); %design matrix for user feedback
            for i=1:num_userfeedback
                S(i,Theta_user(i,2)) = 1;
            end
            sigma_inverse = sigma_inverse + (1/model_params.Nu_user)^2 * S'*S; %part 3
            temp = (1/model_params.Nu_user)^2 * S'*F;
        end

        posterior.sigma = inv(sigma_inverse);
        posterior.mean  = posterior.sigma * ( (1/model_params.Nu_y)^2 * X'*Y + temp );
    end
    %TODO: this posterior.si is required for the sparse case. fix the function interfaces later and remove this line
    posterior.si = [];
end