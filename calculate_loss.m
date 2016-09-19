function [mse,log_pp_test,log_pp_train] = calculate_loss(X_train, Y_train, posterior, X_test, Y_hat, Y_test, model_params)
%Calculate different types of loss/utility functions
    mse = mean((Y_hat- Y_test).^2); %MSE
    %log of posterior predictive dist as the loss function for test data
    post_pred_var = diag(X_test'*posterior.sigma*X_test) + model_params.Nu_y^2;
    log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_test'*posterior.mean - Y_test).^2)./(2*post_pred_var);
    log_pp_test =  mean(log_post_pred);
    %log of posterior predictive dist as the loss function for training data
    post_pred_var = diag(X_train'*posterior.sigma*X_train) + model_params.Nu_y^2;
    log_post_pred = -log(sqrt(2*pi*post_pred_var)) - ((X_train'*posterior.mean - Y_train).^2)./(2*post_pred_var);
    log_pp_train = mean(log_post_pred);
end

