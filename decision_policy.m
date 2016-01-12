function [ selected_feature ] = decision_policy( posterior_samples , Method )
%DECISION_POLICY chooses one of the features to show to the user
    %Method = 1: UCB, 2: random all, 3: random first three
    num_features = size(posterior_samples,2);
    if Method == 1
        UCBs = prctile(posterior_samples,90);
        [~,selected_feature] = max(UCBs);
    end
    if Method == 2 
        selected_feature = ceil(rand*num_features);
    end
    if Method == 3
        selected_feature = ceil(rand*3);
    end
    if Method == 4
        VARs = var(posterior_samples);
        [~,selected_feature]= max(VARs);
    end
    
  
        

end

