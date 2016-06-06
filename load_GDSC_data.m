function [ X_all, Y_all, theta_star ] = load_GDSC_data( drug_number )
%TODO: The data is really weird and I did not understand how to use
%them!!!!!
%LOAD_GDSC_DATA opens and organizes the GDSC data
%The data was organized by Marta and Ammad for the ACML paper. We directly
%use their data check the paper for explanations about the data
%drug_numer should be smaller than 124

    %load the data
    directory_name='DATA_GDSC/';
    load(strcat(directory_name,'input_data/','DrugResponse.mat'));
    Y_all = DrugResponse(:,drug_number);
    
    load(strcat(directory_name,'input_data/','GeneExpression.mat'));
    X_all = GeneExpression;
    
    file = sprintf('LR_model_standardized_alpha_1.00_lamda_CV_Drug_%d.mat',drug_number);
    load(strcat(directory_name,'pseudo_groundtruth/',file));
    
    theta_star = linearReg_results.Beta(drug_number,:)';
    

end

