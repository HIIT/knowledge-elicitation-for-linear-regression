function [ posterior ] = calculate_posterior(X, Y, Theta_user, model_parameters)
% Calculate the posterior based on the analytical formula in the document
    num_features = size(X,1);
    num_userfeedback = size(Theta_user,1);
    posterior = struct('mean',zeros(num_features,1), 'sigma', zeros(num_features,num_features));
    
    sigma_inverse = (1/model_parameters.Nu_theta)^2 * eye(num_features); % part 1
    
    X = X'; %design matrix
    sigma_inverse = sigma_inverse + (1/model_parameters.Nu_y)^2 * X'*X; %part 2 
    
    temp = 0;
    %if user has given feedback
    if num_userfeedback > 0
        F = Theta_user(:,1); % user feedbacks
        S = zeros(num_userfeedback, num_features); %design matrix for user feedback
        for i=1:num_userfeedback
            S(i,Theta_user(i,2)) = 1;
        end
        sigma_inverse = sigma_inverse + (1/model_parameters.Nu_user)^2 * S'*S; %part 3
        temp = (1/model_parameters.Nu_user)^2 * S'*F;
    end
    
    posterior.sigma = inv(sigma_inverse);
    posterior.mean  = posterior.sigma * ( (1/model_parameters.Nu_y)^2 * X'*Y + temp );
    
end