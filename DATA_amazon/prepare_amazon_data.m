clear all


load('amazon_data.mat');
%it gives us X_all_raw, Y_all, keywords_all
appearance_threshold = 20;

num_data     = size(X_all_raw,1);
num_orig_dim = size(X_all_raw,2);

appearance_per_feature = sum(X_all_raw ~= 0,1);
filtered_features = appearance_per_feature> appearance_threshold;
X_all = X_all_raw(:,filtered_features);
keywords = keywords_all(filtered_features);

save('amazon_data','X_all','Y_all','keywords');
disp([' number of data ', num2str(num_data), ', num original dimensions ', num2str(num_orig_dim), ', number of new dimensions ', num2str(size(X_all,2)) ]);