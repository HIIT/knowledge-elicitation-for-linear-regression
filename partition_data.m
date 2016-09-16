function [X_train, X_user, X_test, Y_train, Y_user, Y_test] = partition_data(X_all, Y_all, num_userdata, num_trainingdata)
% Randomly divide the data into training, test, and user data.

    num_data = size(X_all,1);
    
    %randomly select the user data
    userdata_indices  = false(num_data,1);
    selected_userdata = datasample(1:num_data,num_userdata,'Replace',false);
    userdata_indices(selected_userdata) = true;
    X_user   = X_all(userdata_indices,:)'; 
    Y_user   = Y_all(userdata_indices);      
    %randomly select the training data from the remaining data. The rest are test data.
    X_remain = X_all(~userdata_indices,:);
    Y_remain = Y_all(~userdata_indices); 
    train_indices  = false(num_data-num_userdata,1);
    selected_train = datasample(1:num_data-num_userdata,num_trainingdata,'Replace',false);
    train_indices(selected_train) = true;
    test_indices   = ~train_indices;
    
    X_test   = X_remain(test_indices,:)'; 
    Y_test   = Y_remain(test_indices);   
    X_train  = X_remain(train_indices,:)';
    Y_train  = Y_remain(train_indices,:); 
    
end

