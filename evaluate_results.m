clear all
close all

load('results')


num_methods = size(decisions,1);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);
Method_list = char('Max(90% UCB,90% LCB)','Uniformly random','random on the relevelant features','max variance', 'Bayes experiment design');

figure
plot(mean(Loss_1,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value X(theta - theta*)')

figure
plot(mean(Loss_2,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (theta - theta*)')


figure
plot(mean(Loss_3,3)','.-');
legend(Method_list)
title('Loss function')
xlabel('number of expert feedbacks')
ylabel('Loss value (log(posterior predictive))')

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


% legend(Method_list)

% for method =1 : num_methods
%     figure
%     data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
%     hist(data,10)
% end
% 

    