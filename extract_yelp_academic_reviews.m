% source for the YELP academic dataset: https://www.yelp.com/dataset_challenge

fid = fopen('yelp_academic_dataset_review.json');

tline = fgets(fid);
A = {};
numLines = 0;
while ischar(tline)
    numLines = numLines+1;
    A{numLines} = tline;
    tline = fgets(fid);
end
fclose(fid);

% around 2.7 million total reviews...
Max_review_number= 5000;
% randomly selected reviews
rng('default');
rng(1);
p = randperm(numLines-1);
selected_reviews = p(1:Max_review_number);

Max_num_keyword = 70000;

% Create distinct keyword matrix

X_all_raw = zeros(Max_review_number,Max_num_keyword);
Y_all = zeros(Max_review_number,1);
idx_last_key=1;
keywords_all =  cell(1,Max_num_keyword);

for i=1:Max_review_number
    disp(['line number = ', num2str(i), ' num unique keys = ',num2str(idx_last_key)] );
    keywordNumber = strsplit(A{selected_reviews(i)}, ' ');
    % extract the number of stars from column 13 
    a = strsplit(keywordNumber{13}, ',');
    Y_all(i) = str2double(a{1}); 
    % extract the review' text
    for j = 17: (size(keywordNumber,2)-4)
        keyword = keywordNumber{j};  
        number = 1;
        %if this is the first keyword
        if (idx_last_key == 1)           
           keywords_all{idx_last_key} = keyword;
           X_all_raw(i,idx_last_key) = number;
           idx_last_key = idx_last_key + 1;
           continue;
        end
        %check if the keyword already exists
        idx = find(strcmp(keywords_all, keyword));
        if (idx > 0)
            X_all_raw(i,idx) = X_all_raw(i,idx) + 1;
        else
            keywords_all{idx_last_key} = keyword;
            X_all_raw(i,idx_last_key) = number;
            idx_last_key = idx_last_key + 1;
        end          
    end
end
X_all_raw(:,idx_last_key:end) = [];
Y_all(idx_last_key:end)   = [];
keywords_all(idx_last_key:end) = [];

save('yelp_academic_data_raw','X_all_raw','Y_all','keywords_all','-v7.3');


