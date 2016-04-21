clear all
close all

% The current visualization script can only draw the heat map for the case
% when we iterate through the #training_data or #dimensions, assuming the
% other one as a constant value

load('results_all')

num_methods = size(Method_list,2);

num_iterations = size(decisions,2);
num_runs = size(decisions,3);

for loss_function = 1:4     
    if loss_function == 1
        loss = Loss_1;
    end
    if loss_function == 2
        loss = Loss_2;
    end
    if loss_function == 3
        loss = Loss_3;
    end
    if loss_function == 4
        loss = Loss_4;
    end    
    
    if size(num_trainingdata,2) == 1
        %% Assume that the #training_data is fixed and iterate through #dimensions
        t_index = 1;
        figure();
        heat_map = zeros(num_iterations,size(num_features,2));
        for method = 1:num_methods
            for f_index = 1:size(num_features,2)
                cutted_loss = loss(:,:,:,f_index,t_index);
                temp = mean(cutted_loss,3); %average over different runs
                heat_map(:,f_index) = temp(method,:)'; %create a heatmap for each method
            end
            subplot(2,2,method) 
            pcolor(heat_map)
            title(Method_list(method))
            xlabel('number of dimensions')
            ylabel('number of expert feedbacks')
        %     imagesc(heat_map);
        %     colormap(gray)     
        %     colorbar();
        end
        disp(['The number of training data is fixed to ', num2str(num_trainingdata(t_index))])
    end
    
    if size(num_features,2) == 1
        %% Assume that the #dimensions is fixed and iterate through #training_data 
        f_index = 1;
        figure();
        heat_map = zeros(num_iterations,size(num_trainingdata,2));
        for method = 1:num_methods
            for t_index = 1:size(num_trainingdata,2)
                cutted_loss = loss(:,:,:,f_index,t_index); 
                temp = mean(cutted_loss,3); %average over different runs
                heat_map(:,t_index) = temp(method,:)'; %create a heatmap for each method
            end
            subplot(2,2,method) 
            pcolor(heat_map)
            title(Method_list(method))
            xlabel('number of training data')
            ylabel('number of expert feedbacks')          
        %     imagesc(heat_map);
        %     colormap(gray)     
        %     colorbar();
        end
        disp(['The number of dimensions is fixed to ', num2str(num_features(f_index))])
    end   
end


