function [ X_all ] = generate_data( num_data,num_features, normalization_method )
%GENERATE_DATA generates covariates based on some assumptions

    if normalization_method == 1
        % normalization method 1: unit vectors
        X_all = rand(num_data,num_features);% this is all the data
        X_all = X_all./repmat(sqrt(sum(X_all.^2,2)),1,num_features); %normalize X_all into a unit vector
    end
    if normalization_method == 2
        %normalization method 2: spike paper
        X_all = rand(num_data,num_features);% this is all the data
        X_all = X_all./repmat(sqrt(var(X_all)), num_data,1); %var of each feature should be one
        X_all = X_all - repmat(mean(X_all), num_data,1);% data should be zero mean
    end
    if normalization_method == 3
        %normalization method 3: spike paper, alternative
        X_all = mvnrnd(zeros(num_features,1), 0.4*eye(num_features,num_features),num_data);
    end
    
end

