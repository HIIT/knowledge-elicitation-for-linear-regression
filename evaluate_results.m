clear all
close all

load('results')

num_methods = size(Method_list,2);
num_iterations = size(decisions,2);
num_runs = size(decisions,3);

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

figure
Loss_1_mean = mean(Loss_1,3)';
%This one is the data
plot([0:num_iterations-1],Loss_1_mean,'.-','LineWidth',2);
legend(Method_list)
title('Loss function')
xlabel('Number of Expert Feedbacks')
ylabel('Mean Squared Error')


figure
%for real data case Loss_2 is MSE on training in the normalized space
Loss_2_mean = mean(Loss_2,3)';
%This one is the data
plot([0:num_iterations-1], Loss_2_mean,'.-','LineWidth',2);
legend(Method_list)
title('Loss function')
xlabel('Number of Expert Feedbacks')
ylabel('Mean Squared Error on Training')

%divide the decisions in two groups:  0. non-relevant features 1. relevant features
relevants_features = find(z_star == 1);
% non_relevants_features = find(z_star == 0);
binary_decisions = ismember(decisions,relevants_features);
ave_binary_decisions = mean(binary_decisions,3);

figure
plot(ave_binary_decisions','.-','LineWidth',2);
legend(Method_list)
title('Average suggestion behavior of each method')
xlabel('Number of Expert Feedbacks')
ylabel('0 means zero or "do not know" features, 1 means relevant features')

acccumulated_ave_binary_decisions = cumsum(ave_binary_decisions,2);
acccumulated_ave_binary_decisions = [acccumulated_ave_binary_decisions ./ repmat([1:num_iterations],num_methods,1)]';

figure
%This one is the data
plot(acccumulated_ave_binary_decisions,'.-','LineWidth',2);
legend(Method_list)
title('Accumulated average suggestion behavior of each method')
xlabel('Number of Expert Feedbacks')
ylabel('0 means zero or "do not know" features, 1 means relevant features')

%% create tables for comparing number of samples/feedback to reach the same MSE level
MSE_levels = max(max(Loss_1_mean)):-0.01:min(min(Loss_1_mean));
table = zeros(size(MSE_levels,2),num_methods);
for m=1:num_methods
    for mse_ind = 1:size(MSE_levels,2)
        ind = find(Loss_1_mean(:,m) < MSE_levels(mse_ind),1);
        if isempty(ind)
            table(mse_ind,m) = inf;
        else
            table(mse_ind,m) = ind;
        end
    end
end
MSE_table = array2table(table,'VariableNames',regexprep(Method_list,'[^\w'']',''),'RowNames',cellstr(num2str((MSE_levels)')));
%% Plot the three figures in the Finalized format (paper format)
num_ticks = 10;
MarkerIndices = 1:floor(num_iterations/num_ticks):num_iterations;
for fig_num=1:3
    figure
    hold on
    if fig_num == 1
        data = Loss_1_mean;
        xlabel('Number of Expert Feedbacks','FontSize',16)
        ylabel('Mean Squared Error','FontSize',16)
    end
    if fig_num == 2
        data = Loss_2_mean;
        xlabel('Number of Expert Feedbacks','FontSize',16)
        ylabel('Mean Squared Error on Training','FontSize',16)        
    end
    if fig_num == 3
        data = acccumulated_ave_binary_decisions;
        title('Accumulated average suggestion behavior of each method')
        xlabel('Number of Expert Feedbacks','FontSize',16)
        ylabel('0: zero features, 1: non-zero features','FontSize',16)
        %     ylabel('0 means zero or "do not know" features, 1 means relevant features','FontSize',16)
    end
    %Random
    method_ind = find(strcmp('Random', Method_list));
    p1 = plot([0],data(1,method_ind),'-^','LineWidth',2,'MarkerSize',8,'Color',[0,0.5,0]);
    plot(MarkerIndices-1,data(MarkerIndices,method_ind),'^','LineWidth',2,'MarkerSize',8,'Color',[0,0.5,0]);
    plot([0:num_iterations-1],data(:,method_ind),'-','LineWidth',2,'Color',[0,0.5,0]);
    %First relevant features, then non-relevant
    method_ind = find(strcmp('First relevant features, then non-relevant', Method_list));
    p2 = plot([0],data(1,method_ind),'-rv','LineWidth',2,'MarkerSize',8);
    plot(MarkerIndices-1,data(MarkerIndices,method_ind),'rv','LineWidth',2,'MarkerSize',8);
    plot([0:num_iterations-1],data(:,method_ind),'-r','LineWidth',2);
    %Sequential Experimental Design
    method_ind = max([find(strcmp('Expected information gain, full EP approx', Method_list)), ...
        find(strcmp('Expected information gain, fast approx', Method_list))]);
    p3 = plot([0],data(1,method_ind),'-bs','LineWidth',2,'MarkerSize',8);
    plot(MarkerIndices-1,data(MarkerIndices,method_ind),'bs','LineWidth',2,'MarkerSize',8);
    plot([0:num_iterations-1],data(:,method_ind),'-b','LineWidth',2);
    
    if sparse_params.simulated_data == 1
        %non-sequential method (if needed)
        method_ind = max([find(strcmp('Expected information gain, full EP approx, non-sequential', Method_list)), ...
            find(strcmp('Expected information gain, fast approx, non-sequential', Method_list))]);
        p4 = plot([0],data(1,method_ind),'-mo','LineWidth',2,'MarkerSize',8);
        plot(MarkerIndices-1,data(MarkerIndices,method_ind),'mo','LineWidth',2,'MarkerSize',8);
        plot([0:num_iterations-1],data(:,method_ind),'-m','LineWidth',2);
        legend([p1,p2,p3,p4],{'Random','First relevant features, then non-relevant', ...
            'Sequential Experimental Design', 'Non-sequential Experimental Design'},'FontSize',14);
    else
        %we don't need to plot ground truth for simulated data 
        %Ground truth (all feedbacks)
        method_ind = find(strcmp('Ground truth - all feedback', Method_list));
        p4 = plot([0:num_iterations-1],data(:,method_ind),'--k','LineWidth',2);
        legend([p1,p2,p3,p4],{'Random','First relevant features, then non-relevant', ...
            'Sequential Experimental Design', 'Ground truth (all feedbacks)'},'FontSize',14);        
    end
    % legend boxoff
    hold off
end
