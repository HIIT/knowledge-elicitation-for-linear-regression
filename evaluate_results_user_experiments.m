clear all
close all

load('results_user_exp')


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
MSE_without_feedback = mean(mean(Loss_1(num_methods,1,:,1)));
plot([1,num_iterations],MSE_without_feedback*[1,1],'k-');
for user =1:num_users
    plot(mean(Loss_1(:,:,:,user),3)','.-');
end   
legend(['Baseline (No user feedback)', Method_list])
title('Loss function')
xlabel('Number of User Feedback','FontSize',16)
ylabel('Mean Squared Error','FontSize',16)
% ylabel('Loss value (X-test*theta - Y-test)')


ground_truth_all_feedback = find(strcmp('Ground truth - all feedback', Method_list));
if ground_truth_all_feedback
    figure
    hold on
    for user =1:num_users
        ave_loss = mean(Loss_1(:,:,:,user),3);
        percentage_improvement = zeros(num_methods, num_iterations);
        for method =1: num_methods
            all_feedback_gt = ave_loss(ground_truth_all_feedback,1);
%             percentage_improvement(method,:) = 100* (1 - (ave_loss(method,:) - all_feedback_gt)./(ave_loss(method,1) - all_feedback_gt));
            percentage_improvement(method,:) = 100* (ave_loss(method,1) - ave_loss(method,:))./(ave_loss(method,1) - all_feedback_gt);
        end
        percentage_improvement(ground_truth_all_feedback,:) = [];
        plot(percentage_improvement','.-');
    end
    Method_list_temp = Method_list;
    Method_list_temp(ground_truth_all_feedback) = [];
    legend(Method_list_temp)
    title('Loss function')
    xlabel('Number of User Feedback','FontSize',16)
    ylabel('Percentage of Improvememt in MSE','FontSize',16)
    grid on
end
%% check if the results are better than random suggestions
%find random recommender index
method_index_rnd = find(strcmp('Uniformly random', Method_list));
method_index_ours = find(strcmp('Expected information gain (post_pred), fast approx', Method_list));
P_values = zeros(num_iterations,num_users);
CIs = zeros(num_iterations,num_users,2);
hs = zeros(num_iterations,num_users); 
for user =1:num_users
    hold on
    for iteration = 1:num_iterations
        %exteract method results over all runs
        runs_rand = Loss_1(method_index_rnd,iteration,:,user);
        runs_ours = Loss_1(method_index_ours,iteration,:,user);
        runs_rand = reshape(runs_rand,[1,num_runs]);
        runs_ours = reshape(runs_ours,[1,num_runs]);
%         figure
%         hist([runs_ours;runs_rand]'); 
        %[h,p,ci,stats] = ttest2(runs_rand,runs_ours);
        [h,p,ci,stats] = ttest(runs_rand,runs_ours);
        P_values(iteration,user) = p;
        CIs(iteration,user,:) = ci;
        hs(iteration,user) = h;
    end
end
figure
plot(P_values,'.-');
title('Two-sample t-test for each participants (random vs Info. gain suggestions)')
xlabel('number of expert feedbacks')
ylabel('P-value')

[~, significant_thrsholds] = max(hs);
disp(['Difference between metods is significant after ', num2str(max(significant_thrsholds)-1),' feedback.']);

figure
plot(-log10(P_values),'.-');
title('Two-sample t-test for each participants (random vs Info. gain suggestions) (5% significance level)')
xlabel('number of expert feedbacks')
ylabel('-Log 10 (P-value)')

figure
hold on
plot(CIs(:,:,1));
plot(CIs(:,:,2));
plot([1,num_iterations],[0,0],'k-');
hold off
title('Confidence interval for the mean of MSE of (random - Info. gainx) for the paired t-test')
xlabel('number of expert feedbacks')
ylabel('random - Info. gain')
%% 


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
ylabel('0 means not-relevant or "do not know" features, 1 means relevant features')

figure
hold on
for user =1:num_users
    ave_binary_decisions = mean(binary_decisions(:,:,:,user),3);
    acccumulated_ave_binary_decisions = cumsum(ave_binary_decisions,2);
    acccumulated_ave_binary_decisions = acccumulated_ave_binary_decisions ./ repmat([1:num_iterations],num_methods,1);
%     acccumulated_ave_binary_decisions(ground_truth_all_feedback,:) = [];
    plot(acccumulated_ave_binary_decisions','.-');   
end
% legend(Method_list_temp)
legend(Method_list)
title('Accumulated average suggestion behavior of each method')
xlabel('Number of User Feedback')
ylabel('0 means "not-relevant" or "uncertain" keywords, 1 means "relevant" keywords')

% %show the histogram of decisions
% for method =1 : num_methods
%     figure
%     data = reshape(decisions(method,:,:),[num_iterations*num_runs,1]);
%     hist(data,num_features)
% end


    
