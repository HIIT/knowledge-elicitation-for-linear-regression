%% Initialize variables.
filename = 'fb_160816_all.csv';
delimiter = ',';

%% Format string for each line of text:
%   column1: text (%s)
%	column2: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Allocate keyword names and their relevance feedback
feedback_key = dataArray{:, 1};
feedback_val = dataArray{:, 2};
%% load other xlsc files
temp = xlsread('textData_Y.xlsx');
Y_all = temp(:,2);
Y_all_doc = temp(:,1);

[~, ~, X_all_key] = xlsread('C:\Matlab Projects\PhD project\Prior elicitation\CanDI_text_data\textData_X.xlsx','Sheet1','B1:QP1');
X_all_key(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),X_all_key)) = {''};
temp = xlsread('textData_X.xlsx');
X_all_doc = temp(:,1);
X_all = temp(:,2:end);
%% Create necessary variables for the main simulation

z_star = zeros(1,size(X_all_key,2)); %expert relevant/non-relevant feedback
for i=1: size(X_all_key,2)
    for j=1: size(feedback_key,1)
        if strcmp(X_all_key{i},feedback_key{j})
            z_star(i) = feedback_val(j);
            break    
        end
    end
end

save('candi_data', 'X_all','X_all_key','X_all_doc','Y_all','z_star');