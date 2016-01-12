function [ log_posterior ] = posterior_evaluation(theta, X, Y, Theta_user, model_parameters)
%POSTERIOR_EVALUATION evalutes the unnormalized log-posterior at point theta 
%The posterior formula is shown in EQ X in the document
    likelihood_ys = normpdf(Y, X'*theta, model_parameters.Nu_y);    
    prior = mvnpdf(theta', zeros(1,size(theta,1)), model_parameters.Nu_theta^2 * eye(size(theta,1)) );
    log_posterior = log(prior) + sum(log(likelihood_ys));
    if size(Theta_user)>0
        likelihood_users = normpdf(Theta_user(:,1), theta(Theta_user(:,2)), model_parameters.Nu_user);
        log_posterior = log_posterior + sum(log(likelihood_users));
    end

end

