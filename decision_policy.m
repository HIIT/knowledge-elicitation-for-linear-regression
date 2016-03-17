function [ selected_feature ] = decision_policy( posterior , Method, num_nonzero_features )
%DECISION_POLICY chooses one of the features to show to the user
    %Method = 1: UCB, 2: random all, 3: random first three
    num_features = size(posterior.mean,1);
    if Method == 1 %Combination of UCB and LCB
        % Assume that the features of Theta are independent
        % use 0.9 percentile for now       
        UBs = abs(posterior.mean) + 1.28155 * sqrt(diag(posterior.sigma));
        [~,selected_feature] = max(UBs);
    end
    if Method == 2 %randomly choose one feature
        selected_feature = ceil(rand*num_features);
    end
    if Method == 3 %randomly choose one of the nonzero features
        selected_feature = ceil(rand*num_nonzero_features);
    end
    if Method == 4 %choose the feature with highest posterior variance
        %Assume that features of Theta are independent
        VARs = diag(posterior.sigma);
        [~,selected_feature]= max(VARs);
    end
             

end

