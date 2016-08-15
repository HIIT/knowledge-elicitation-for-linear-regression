clear all
close all

load('results')


num_methods = size(Method_list,2);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);

%display useful information about the simulation
disp(struct2table(model_params));
disp(['Number of features: ', num2str(num_features),'.']);
disp(['Number of non-zero features: ', num2str(num_nonzero_features),'.']);
disp(['Number of training data: ', num2str(num_trainingdata),'.']);
disp(['Data normalization method: ', num2str(normalization_method),'.']);
disp(['Averaged over ', num2str(num_runs), ' runs.']);


figure
plot(mean(Loss_1,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (X-test*theta - Y-test)')

figure
plot(mean(Loss_2,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (theta - theta*)')


figure
plot(mean(Loss_3,3)','.-');
legend(Method_list)
title('Utility function')
xlabel('number of expert feedbacks')
ylabel('Utility value (log(posterior predictive))')

figure
plot(mean(Loss_4,3)','.-');
legend(Method_list)
title('Utility function')
xlabel('number of expert feedbacks')
ylabel('Utility value (log(posterior predictive)) on tr.data')

%divide the decisions in two groups:  0. features with zero values 1. features with non-zero values
binary_decisions = decisions <= num_nonzero_features;
ave_binary_decisions = mean(binary_decisions,3);

figure
plot(ave_binary_decisions','.-');
legend(Method_list)
title('Average suggestion behavior of each method')
xlabel('number of expert feedbacks')
ylabel('0 means zero features, 1 means non-zero features')

acccumulated_ave_binary_decisions = cumsum(ave_binary_decisions,2);
acccumulated_ave_binary_decisions = acccumulated_ave_binary_decisions ./ repmat([1:num_iterations],num_methods,1);

figure
plot(acccumulated_ave_binary_decisions','.-');
legend(Method_list)
title('Accumulated average suggestion behavior of each method')
xlabel('number of expert feedbacks')
ylabel('0 means zero features, 1 means non-zero features')

% %show the histogram of decisions
% for method =1 : num_methods
%     figure
%     data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
%     hist(data,num_features)
% end


    
