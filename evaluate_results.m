clear all
close all

load('results')


num_methods = size(Method_list,2);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);

%display useful information about the simulation
disp(struct2table(model_params));
% disp(struct2table(sparse_params));
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
Loss_1_mean = mean(Loss_1,3)';
hold on
% %This one is for the legend, it should be removed after fixing the legends
% ticks = 20;
% plot(1:ticks:num_iterations,Loss_1_mean(1:ticks:end,:),'-s','LineWidth',2);
% % legend boxoff
% %This one is for the markers
% plot(1:ticks:num_iterations,Loss_1_mean(1:ticks:end,:),'s','LineWidth',2);
%This one is the data
plot(Loss_1_mean,'.-','LineWidth',2);
legend(Method_list)
hold off
title('Loss function')
xlabel('Number of Expert Feedbacks','FontSize',16)
ylabel('Mean Squared Error','FontSize',16)
% ylabel('Loss value (X-test*theta - Y-test)','FontSize',16)
% legend({'GT','Random', 'Random on the relevant features', ...
%     'Sequential Experimental Design', 'Non-sequential Experimental Design'},'FontSize',14)

figure
plot(mean(Loss_2,3)','.-','LineWidth',2);
legend(Method_list)
title('Utility function')
xlabel('Number of Expert Feedbacks','FontSize',16)
ylabel('Utility value (log(posterior predictive))','FontSize',16)

figure
plot(mean(Loss_3,3)','.-','LineWidth',2);
legend(Method_list)
title('Utility function')
xlabel('Number of Expert Feedbacks','FontSize',16)
ylabel('Utility value (log(posterior predictive)) on tr.data','FontSize',16)


if exist('Loss_4')
    figure
    Loss_4_mean = mean(Loss_4,3)';
    hold on
%     %This one is for the legend, it should be removed after fixing the legends
%     plot(1:ticks:num_iterations,Loss_4_mean(1:ticks:end,:),'-s','LineWidth',2);
%     % legend boxoff
%     %This one is for the markers
%     plot(1:ticks:num_iterations,Loss_4_mean(1:ticks:end,:),'s','LineWidth',2);
    %This one is the data
    plot(Loss_4_mean,'.-','LineWidth',2);
    legend(Method_list)
    hold off
    title('Loss function')
    xlabel('Number of Expert Feedbacks','FontSize',16)
    ylabel('Mean Squared Error on Training','FontSize',16)
%     legend({'GT','Random', 'Random on the relevant features', ...
%         'Sequential Experimental Design', 'Non-sequential Experimental Design'},'FontSize',14)
end

%divide the decisions in two groups:  0. non-relevant features 1. relevant features
relevants_features = find(z_star == 1);
% non_relevants_features = find(z_star == 0);
binary_decisions = ismember(decisions,relevants_features);
ave_binary_decisions = mean(binary_decisions,3);

figure
plot(ave_binary_decisions','.-','LineWidth',2);
legend(Method_list)
title('Average suggestion behavior of each method')
xlabel('Number of Expert Feedbacks','FontSize',16)
ylabel('0 means zero or "do not know" features, 1 means relevant features','FontSize',16)

acccumulated_ave_binary_decisions = cumsum(ave_binary_decisions,2);
acccumulated_ave_binary_decisions = [acccumulated_ave_binary_decisions ./ repmat([1:num_iterations],num_methods,1)]';

figure
hold on
% %This one is for the legend, it should be removed after fixing the legends
% plot(1:ticks:num_iterations,acccumulated_ave_binary_decisions(1:ticks:end,:),'-s','LineWidth',2);
% % legend boxoff
% %This one is for the markers
% plot(1:ticks:num_iterations,acccumulated_ave_binary_decisions(1:ticks:end,:),'s','LineWidth',2);
%This one is the data
plot(acccumulated_ave_binary_decisions,'.-','LineWidth',2);
legend(Method_list)
% legend({'GT','Random', 'Random on the relevant features', ...
%     'Sequential Experimental Design', 'Non-sequential Experimental Design'},'FontSize',14)
title('Accumulated average suggestion behavior of each method')
xlabel('Number of Expert Feedbacks','FontSize',16)
ylabel('0 means zero or "do not know" features, 1 means relevant features','FontSize',16)

% %show the histogram of decisions
% for method =1 : num_methods
%     figure
%     data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
%     hist(data,num_features)
% end


    
%SAVE AS eps