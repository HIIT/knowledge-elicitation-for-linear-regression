close all
clear all
%The following line loads X_all, Y_all
% load('DATA_Genome\mutation_outputs');
% load('DATA_Genome\mutation_and_cyto_outputs');
load('results_genome.mat');

%% for the original space
figure
imagesc(Loss_1);
averag_mse_all = mean(mean(Loss_1))
figure
imagesc(Loss_4);

figure
subplot(2,1,1) 
boxplot(Y_all',drug_names);
ylabel('Drug responses')
subplot(2,1,2) 
boxplot(Loss_1',drug_names);
ylabel('LOO MSE')

% Correlation between drugs
[RHO_P,PVAL_P] = corr(Y_all,Y_hat_all,'Type','Spearman');
disp('patient-wise correlation')
diag(RHO_P)
[RHO_D,PVAL_D] = corr(Y_all',Y_hat_all','Type','Spearman');
disp('drug-wise correlation')
diag(RHO_D)
%% for the normalized space
figure
imagesc(Loss_1_norm);
averag_mse_all_norm = mean(mean(Loss_1_norm))
figure
imagesc(Loss_4_norm);

figure
subplot(2,1,1) 
boxplot(Y_all_norm',drug_names);
ylabel('Drug responses (normalized space)')
subplot(2,1,2) 
boxplot(Loss_1_norm',drug_names);
ylabel('LOO MSE (normalized space)')

% Correlation between drugs
[RHO_P,PVAL_P] = corr(Y_all_norm,Y_hat_norm,'Type','Spearman');
disp('patient-wise correlation')
diag(RHO_P)
[RHO_D,PVAL_D] = corr(Y_all_norm',Y_hat_norm','Type','Spearman');
disp('drug-wise correlation')
diag(RHO_D)


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