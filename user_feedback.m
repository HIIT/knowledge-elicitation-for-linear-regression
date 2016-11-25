function [ feedback_value ] = user_feedback(feature_index, theta_star, z_star, MODE, sparse_params)
% Generate user feedbacks based on the MODE of the simulation and the model parameters
% Inputs:
% MODE            Feedback type: 1: noisy observation of weight. 2: binary relevance of feature
% feature_index   index of the feature that receives feedback
% theta_star      true weight values
% z_star          true values for the latent variable in spike and slab model
% simulation      true: the user has been generated, false: real data 

    if MODE == 1
        %in case we are using simulated data then use the model parameters
        if sparse_params.simulated_data
            %user feedback is a noisy observation of the weight value
            feedback_value = normrnd(theta_star(feature_index), sqrt(sparse_params.eta2));
        else
            %if we are using real data, use the user feedback (do not add
            %the model noise since data has noise itself)
            feedback_value = theta_star(feature_index);
        end
    end
    
    if MODE == 2
        %user feedback is on the relevance of the weight value
        %In some cases the user may not know the relevance (too much
        %uncertainty). Use -1 to represent "don't know" responses.
        if z_star(feature_index) == -1
            feedback_value = -1;
            return
        end
        %in case we are using simulated data then use the model parameters
        if sparse_params.simulated_data
            f_is_correct = binornd(1,sparse_params.p_u);
            if f_is_correct == 1
                feedback_value = z_star(feature_index);
            else
                feedback_value = ~z_star(feature_index);
            end
        else
            %if we are using real data, use the user feedback (do not add
            %the model noise since data has noise itself)
            feedback_value = z_star(feature_index);
        end
    end

end

