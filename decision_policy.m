function [ selected_feature ] = decision_policy( posterior_samples )
%DECISION_POLICY chooses one of the features to show to the user
    num_features = size(posterior_samples,2);
    
    %.....
    % TO DO: take the first three futures which are relevant
    selected_feature = ceil(rand*num_features);

end

