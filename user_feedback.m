function [ feedback_value ] = user_feedback(feature_index, theta_star, z_star, MODE, model_params)
% Generate user feedbacks based on the MODE of the simulation and the model parameters
% Inputs:
% MODE            Feedback type: 0 and 1: noisy observation of weight. 2: binary relevance of feature
% feature_index   index of the feature that receives feedback
% theta_star      true weight values
% z_star          true values for the latent variable in spike and slab model

    if MODE == 0 || MODE == 1
        %user feedback is a noisy observation of the weight value
        feedback_value = normrnd(theta_star(feature_index),model_params.Nu_user);
    end
    if MODE == 2
        %user feedback is on relevance of the weight value
        f_is_correct = binornd(1,model_params.P_user);
        if f_is_correct == 1
            feedback_value = z_star(feature_index);
        else
            feedback_value = ~z_star(feature_index);
        end
    end

end

