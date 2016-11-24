close all
clear all
load('results_genome.mat');

num_methods = size(Loss_1,1);
num_drugs   = size(Loss_1,2); 
num_data    = size(Loss_1,3);

mse_average = mean(Loss_1,3);
method_names = {'Naive','SS_No_FB','SS_FB_Caroline','SS_FB_Mamun','Ridge'};
MSE_original = array2table(mse_average','VariableNames',method_names,'RowNames',selected_drug_names')
% writetable(MSE_original,'MSE_original','WriteRowNames',true)
%remove the following line later (it is needed because of a small bug)
Loss_1_norm = Loss_1_norm(1:num_methods,1:num_drugs,1:num_data);
mse_average_norm = mean(Loss_1_norm,3);
MSE_norm = array2table(mse_average_norm','VariableNames',method_names,'RowNames',selected_drug_names')
% writetable(MSE_norm,'MSE_norm','WriteRowNames',true)

% for drug=1:num_drugs
%     figure
%     loss_1 = reshape(Loss_1(:,drug,:), [num_methods,num_data,]);
%     boxplot(loss_1',method_names);  
% end

%% perform paired test: Does extra feed help (compared to no feedback)?
MSE_SS_no_FB = reshape(Loss_1(2,:,:), [num_drugs,num_data,]);
MSE_SS_FB_C  = reshape(Loss_1(3,:,:), [num_drugs,num_data,]);
MSE_SS_FB_M  = reshape(Loss_1(4,:,:), [num_drugs,num_data,]);

[h_c,p_c,ci_c,stats_c] = ttest(MSE_SS_no_FB',MSE_SS_FB_C');
[h_m,p_m,ci_m,stats_m] = ttest(MSE_SS_no_FB',MSE_SS_FB_M');

MSE_SS_no_FB_norm = reshape(Loss_1_norm(2,:,:), [num_drugs,num_data,]);
MSE_SS_FB_C_norm  = reshape(Loss_1_norm(3,:,:), [num_drugs,num_data,]);
MSE_SS_FB_M_norm  = reshape(Loss_1_norm(4,:,:), [num_drugs,num_data,]);

[h_c_n,p_c_n,ci_c_n,stats_c_n] = ttest(MSE_SS_no_FB_norm',MSE_SS_FB_C_norm');
[h_m_n,p_m_n,ci_m_n,stats_m_n] = ttest(MSE_SS_no_FB_norm',MSE_SS_FB_M_norm');


%%
% % % %% for the original space
% % % figure
% % % imagesc(Loss_1);
% % % averag_mse_all = mean(mean(Loss_1))
% % % % figure
% % % % imagesc(Loss_4);
% % % 
% % % average_mse_naive = mean(mean(Loss_naive))
% % % 
% % % figure
% % % subplot(3,1,1) 
% % % boxplot(Y_all',drug_names);
% % % ylabel('Drug responses')
% % % subplot(3,1,2) 
% % % boxplot(Y_hat_all',drug_names);
% % % ylabel('Drug predictions')
% % % subplot(3,1,3) 
% % % boxplot(Loss_1',drug_names);
% % % ylabel('LOO MSE')
% % % 
% % % % Correlation between drugs
% % % [RHO_P,PVAL_P] = corr(Y_all,Y_hat_all,'Type','Spearman');
% % % disp('patient-wise correlation')
% % % diag(RHO_P)
% % % [RHO_D,PVAL_D] = corr(Y_all',Y_hat_all','Type','Spearman');
% % % disp('drug-wise correlation')
% % % diag(RHO_D)
% % % 
% % % [RHO_P,PVAL_P] = corr(Y_all,Y_naive,'Type','Spearman');
% % % disp('patient-wise correlation NAIVE')
% % % diag(RHO_P)
% % % [RHO_D,PVAL_D] = corr(Y_all',Y_naive','Type','Spearman');
% % % disp('drug-wise correlation NAIVE')
% % % diag(RHO_D)
% % % %% for the normalized space
% % % figure
% % % imagesc(Loss_1_norm);
% % % averag_mse_all_norm = mean(mean(Loss_1_norm))
% % % % figure
% % % % imagesc(Loss_4_norm);
% % % 
% % % average_mse_naive_norm = mean(mean(Loss_naive_norm))
% % % 
% % % 
% % % figure
% % % subplot(3,1,1) 
% % % boxplot(Y_all_norm',drug_names);
% % % ylabel('Drug responses (normalized space)')
% % % subplot(3,1,2) 
% % % boxplot(Y_hat_norm',drug_names);
% % % ylabel('Drug predictions (normalized space)')
% % % subplot(3,1,3)
% % % boxplot(Loss_1_norm',drug_names);
% % % ylabel('LOO MSE (normalized space)')
% % % 
% % % % Correlation between drugs
% % % [RHO_P,PVAL_P] = corr(Y_all_norm,Y_hat_norm,'Type','Spearman');
% % % disp('patient-wise correlation')
% % % diag(RHO_P)
% % % [RHO_D,PVAL_D] = corr(Y_all_norm',Y_hat_norm','Type','Spearman');
% % % disp('drug-wise correlation')
% % % diag(RHO_D)
% % % 
% % % [RHO_P,PVAL_P] = corr(Y_all_norm,Y_naive_norm,'Type','Spearman');
% % % disp('patient-wise correlation NAIVE')
% % % diag(RHO_P)
% % % [RHO_D,PVAL_D] = corr(Y_all_norm',Y_naive_norm','Type','Spearman');
% % % disp('drug-wise correlation NAIVE')
% % % diag(RHO_D)

% close all
% clear all
% drug_names = {'AZD2014','Dactolisib','Dexamethasone','Idelalisib','Methylprednisolone',...
%     'PF.04691502','Pictilisib','Pimasertib','Refametinib','Temsirolimus','Trametinib','Venetoclax'};
% drug_indices= [5  12  14  20  24  29  30  31  33  36  38  40];
% %The following line loads X_all, Y_all
% % load('DATA_Genome\mutation_outputs');
% load('DATA_Genome\mutation_and_cyto_outputs');
% %% Visualize the data 
% visualize_data = false;
% if visualize_data
%     boxplot(Y_all,'PlotStyle','compact');
%     figure
%     imagesc(Y_all');
%     
%     [pc,~,latent,~] = princomp(X_all);
%     X_all_pc = X_all * pc(:,1:2);
%     figure
%     plot(X_all_pc(:,1),X_all_pc(:,2),'.')
% end
% 
% load('results_genome_learn params.mat');
% figure
% imagesc(Loss_1);
% averag_mse_all = mean(mean(Loss_1));
% 
% figure
% imagesc(Loss_1(drug_indices,:));
% 
% figure
% subplot(2,1,1) 
% boxplot(Y_all(:,drug_indices),drug_names);
% ylabel('Drug responses')
% subplot(2,1,2) 
% boxplot(Loss_1(drug_indices,:)',drug_names);
% ylabel('LOO MSE')
% 
% averag_mse_subset = mean(mean(Loss_1(selected_drug_indices,:)));