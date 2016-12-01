clear all
close all

load('results_user_exp')

num_methods = size(Method_list,2);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);
num_users = size(decisions,4);

%display useful information about the simulation
disp(struct2table(sparse_params));
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
disp(['Averaged over ', num2str(num_runs), ' runs.']);

%MSE per method per user
colorstring = 'kgbrym';
figure
hold on
MSE_without_feedback = mean(mean(Loss_1(num_methods,1,:,1)));
plot([0,num_iterations-1],MSE_without_feedback*[1,1],'k--');
for user =1:num_users
    Loss_1_user = mean(Loss_1(:,:,:,user),3);
    for method = 1:num_methods
        plot([0:num_iterations-1],Loss_1_user(method,:)','.-','Color', colorstring(method));
    end
end   
legend(['Baseline (No user feedback)', Method_list])
title('Loss function')
xlabel('Number of User Feedbacks','FontSize',16)
ylabel('Mean Squared Error','FontSize',16)

%percentage of MSE improvement as defined in the paper
method_index_gt = find(strcmp('Ground truth - all feedback', Method_list));
if method_index_gt
    figure
    hold on
    for user =1:num_users
        Loss_1_user = mean(Loss_1(:,:,:,user),3);
        percentage_improvement = zeros(num_methods, num_iterations);
        for method =1: num_methods
            if method ~= method_index_gt
                all_feedback_gt = Loss_1_user(method_index_gt,1);
                percentage_improvement(method,:) = 100* (Loss_1_user(method,1) - Loss_1_user(method,:))./(Loss_1_user(method,1) - all_feedback_gt);
                plot([0:num_iterations-1],percentage_improvement(method,:)','.-','Color', colorstring(method));
            end
        end
    end
    Method_list_temp = Method_list;
    Method_list_temp(method_index_gt) = [];
    legend(Method_list_temp)
    title('Loss function')
    xlabel('Number of User Feedbacks','FontSize',16)
    ylabel('Percentage of Improvememt in MSE','FontSize',16)
    grid on
    hold off
end
%% Statistical tests, check if the results are better than random suggestions
%find the indices of random recommender and out method
method_index_rnd = find(strcmp('Random', Method_list));
method_index_ED = find(strcmp('Expected information gain, fast approx', Method_list));
%this is the old statitistical test. Sami said it is no good!
% P_values = zeros(num_iterations,num_users);
% CIs = zeros(num_iterations,num_users,2);
% hs = zeros(num_iterations,num_users); 
% for user =1:num_users
%     hold on
%     for iteration = 1:num_iterations
%         %exteract method results over all runs
%         runs_rand = Loss_1(method_index_rnd,iteration,:,user);
%         runs_ours = Loss_1(method_index_ED,iteration,:,user);
%         runs_rand = reshape(runs_rand,[1,num_runs]);
%         runs_ours = reshape(runs_ours,[1,num_runs]);
% %         figure
% %         hist([runs_ours;runs_rand]'); 
%         %[h,p,ci,stats] = ttest2(runs_rand,runs_ours);
%         [h,p,ci,stats] = ttest(runs_rand,runs_ours);
%         P_values(iteration,user) = p;
%         CIs(iteration,user,:) = ci;
%         hs(iteration,user) = h;
%     end
% end

%% This part is about the new statistical test that Tomi suggested. 
% We are only checking the difference between mean users 
Ave_users_rand = zeros(num_iterations,num_users);
Ave_users_ED = zeros(num_iterations,num_users);

for user =1:num_users   
    ave_over_runs= mean(Loss_1(:,:,:,user),3);
    Ave_users_ED(:,user) = ave_over_runs(method_index_ED,:);
    Ave_users_rand(:,user) = ave_over_runs(method_index_rnd,:);
end
P_values = zeros(num_iterations,1);
CIs = zeros(num_iterations,2);
hs = zeros(num_iterations,1); 
for iteration = 1:num_iterations
    %exteract method results over all runs
    [h,p,ci,stats] = ttest(Ave_users_rand(iteration,:),Ave_users_ED(iteration,:), 'Alpha',0.05/num_iterations);
    P_values(iteration) = p;
    CIs(iteration,:) = ci;
    hs(iteration) = h;
end
[~, significant_thrsholds] = max(hs)

figure
plot([0:num_iterations-1],P_values,'.-');
title('Two-sample t-test for each participants (random vs Info. gain suggestions)')
xlabel('number of expert feedbacks')
ylabel('P-value')

[~, significant_thrsholds] = max(hs);
disp(['Difference between metods is significant after ', num2str(max(significant_thrsholds)-1),' feedback.']);
disp('Warning: Please also check hs vector since for some iterations (the first one) P_values can be NAN.');

figure
plot([0:num_iterations-1],-log10(P_values),'.-');
title('Two-sample t-test for each participants (random vs Info. gain suggestions) (5% significance level)')
xlabel('number of expert feedbacks')
ylabel('-Log 10 (P-value)')

figure
hold on
plot([0:num_iterations-1],CIs(:,1),'r');
plot([0:num_iterations-1],CIs(:,2),'r');
plot([0,num_iterations-1],[0,0],'k-');
hold off
title('Confidence interval for the mean of MSE of (random - Info. gainx) for the paired t-test')
xlabel('number of expert feedbacks')
ylabel('random - Info. gain')
%% 
%MSE on training (normalized) per method per user
figure
hold on
for user =1:num_users
    Loss_2_user = mean(Loss_2(:,:,:,user),3);
    for method = 1:num_methods
        plot([0:num_iterations-1],Loss_2_user(method,:)','.-','Color', colorstring(method));
    end
end
legend(Method_list)
title('Loss function')
xlabel('Number of User Feedbacks','FontSize',16)
ylabel('Mean Squared Error on training data (normalized)','FontSize',16)


%divide the decisions in two groups:  0. non-relevant features 1. relevant features
relevants_features = find(z_star == 1);
% non_relevants_features = find(z_star == 0);
binary_decisions = ismember(decisions,relevants_features);


figure
hold on
for user =1:num_users
    ave_binary_decisions = mean(binary_decisions(:,:,:,user),3);
    for method = 1:num_methods
        if method ~= method_index_gt       
            plot(ave_binary_decisions(method,:)','.-','Color', colorstring(method));
        end
    end          
end
Method_list_temp = Method_list;
Method_list_temp(method_index_gt) = [];
legend(Method_list_temp)
title('Average suggestion behavior of each method')
xlabel('Number of expert feedbacks','FontSize',16)
ylabel('0 means not-relevant or "do not know" features, 1 means relevant features','FontSize',16)

figure
hold on
for user =1:num_users
    ave_binary_decisions = mean(binary_decisions(:,:,:,user),3);
    acccumulated_ave_binary_decisions = cumsum(ave_binary_decisions,2);
    acccumulated_ave_binary_decisions = acccumulated_ave_binary_decisions ./ repmat([1:num_iterations],num_methods,1);
%     acccumulated_ave_binary_decisions(ground_truth_all_feedback,:) = [];
%     plot(acccumulated_ave_binary_decisions','.-');   
    for method = 1:num_methods
        if method ~= method_index_gt       
            plot(acccumulated_ave_binary_decisions(method,:)','.-','Color', colorstring(method));
        end
    end     
end
Method_list_temp = Method_list;
Method_list_temp(method_index_gt) = [];
legend(Method_list_temp)
title('Accumulated average suggestion behavior of each method')
xlabel('Number of User Feedbacks','FontSize',16)
ylabel('0 means "not-relevant" or "uncertain" keywords, 1 means "relevant" keywords','FontSize',16)
   
