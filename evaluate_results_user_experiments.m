clear all
close all

load('results')


num_methods = size(Method_list,2);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);
num_users = size(decisions,4);

%display useful information about the simulation
disp(struct2table(model_params));
if MODE == 1
    disp('Feedback is on the weight value of features');
end
if MODE == 2
    disp('Feedback is on the relevance of features');
end
disp(['Number of features: ', num2str(num_features),'.']);
disp(['Number of "relevant" features: ', num2str(sum(z_star==1)),'.']);
disp(['Number of "do not know" features: ', num2str(sum(z_star==-1)),'.']);
disp(['Number of training data: ', num2str(num_trainingdata),'.']);
disp(['Data normalization method: ', num2str(normalization_method),'.']);
disp(['Averaged over ', num2str(num_runs), ' runs.']);


figure
hold on
for user =1:num_users
    plot(mean(Loss_1(:,:,:,user),3)','.-');
end   
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (X-test*theta - Y-test)')

% % figure
% % plot(mean(Loss_2,3)','.-');
% % legend(Method_list)
% % title('Loss function')
% % xlabel('number of expert feedbacks')
% % ylabel('Loss value (theta - theta*)')


figure
hold on
for user =1:num_users
    plot(mean(Loss_2(:,:,:,user),3)','.-');
end
legend(Method_list)
title('Utility function')
xlabel('number of expert feedbacks')
ylabel('Utility value (log(posterior predictive))')

figure
hold on
for user =1:num_users
    plot(mean(Loss_3(:,:,:,user),3)','.-');
end
legend(Method_list)
title('Utility function')
xlabel('number of expert feedbacks')
ylabel('Utility value (log(posterior predictive)) on tr.data')

%divide the decisions in two groups:  0. non-relevant features 1. relevant features
relevants_features = find(z_star == 1);
% non_relevants_features = find(z_star == 0);
binary_decisions = ismember(decisions,relevants_features);


figure
hold on
for user =1:num_users
    ave_binary_decisions = mean(binary_decisions(:,:,:,user),3);
    plot(ave_binary_decisions','.-');    
end
legend(Method_list)
title('Average suggestion behavior of each method')
xlabel('number of expert feedbacks')
ylabel('0 means zero or "do not know" features, 1 means relevant features')

figure
hold on
for user =1:num_users
    ave_binary_decisions = mean(binary_decisions(:,:,:,user),3);
    acccumulated_ave_binary_decisions = cumsum(ave_binary_decisions,2);
    acccumulated_ave_binary_decisions = acccumulated_ave_binary_decisions ./ repmat([1:num_iterations],num_methods,1);
    plot(acccumulated_ave_binary_decisions','.-');   
end
legend(Method_list)
title('Accumulated average suggestion behavior of each method')
xlabel('number of expert feedbacks')
ylabel('0 means zero or "do not know" features, 1 means relevant features')

% %show the histogram of decisions
% for method =1 : num_methods
%     figure
%     data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
%     hist(data,num_features)
% end


    
