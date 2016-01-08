function [ posterior_samples ] = sample_posterior(X, Y, Theta_user, model_parameters)
%SAMPLE_POSTERIOR generates samples from the posterior distribution by
%using MCMC (Metropolis) algorithm. 

    %% Parameters
    num_output_samples = 200;
    N = 25000; %chain lenght
    num_MC = 3; %number of chains
    num_features = size(X,1);
    % initialization
    MCs = zeros(N,num_features,num_MC); % array to save the chains 

    %% MCMC ALGORITHM
    
    for chain_num = 1:num_MC
        rejection_rate = 0;
        % Set the starting values for parameters, here theta 
%         theta = 0.5*randn(num_features,1);
        theta = rand(num_features,1);
        
        % Evaluate the log-posterior at starting value
        E_old = posterior_evaluation(theta, X, Y, Theta_user, model_parameters);

        % Metropolis-Hastings algorithm here
        for t=1:N   
            MCs(t,:,chain_num) = theta;
            % jumping distribution: symmetric Normal dis with mean on previous theta and cov=I
            theta_new = mvnrnd(theta,0.05*eye(num_features));
            E_new = posterior_evaluation(theta_new', X, Y, Theta_user, model_parameters);
            r = exp( E_new - E_old);
            if rand< min(r,1)
                theta = theta_new;
                E_old = E_new;
                rejection_rate = rejection_rate + 1;
            end    
        end
        rejection_rate = (N-rejection_rate)/N
    end
    
    %remove the first half of the chains (burn-in)
    MCs(1:N/2,:,:) = [];
    %convergenec analysis       
    [R,neff,Vh,W,B,tau] = psrf(MCs);
    R
    thinning = 1:ceil(num_MC*size(MCs,1)/num_output_samples):size(MCs,1);
    posterior_samples = [];
    for i=1:num_MC
        posterior_samples = [posterior_samples; MCs(thinning,:,i)];
    end
end

