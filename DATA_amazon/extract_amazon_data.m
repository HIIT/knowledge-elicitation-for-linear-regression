%% data source: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
%% [processed_stars.tar.gz] (33 M) 
%% link to related paper using the data: http://link.springer.com/article/10.1007/s10994-014-5475-7 


fid = fopen('all_balanced.review');

tline = fgets(fid);
A = {};
numLines = 0;
while ischar(tline)
    numLines = numLines+1;
    A{numLines} = tline;
    tline = fgets(fid);
end
fclose(fid);

Max_num_keyword = 192676;

X_all_raw = zeros(numLines,Max_num_keyword);
Y_all = zeros(numLines,1);
idx_last_key=1;
keywords_all =  cell(1,Max_num_keyword);
for i=1:numLines
    disp(['line number = ', num2str(i), ' num unique keys = ',num2str(idx_last_key)] );
    keywordNumber = strsplit(A{i}, ' ');
    for j = 1: size(keywordNumber,2)
        a = strsplit(keywordNumber{j}, ':');
        keyword = a{1};  
        number = 0;
        if (size(a, 2) == 2)
            number = str2double(a{2});
        end
        %if this is the first keyword
        if (idx_last_key == 1)           
           keywords_all{idx_last_key} = keyword;
           X_all_raw(i,idx_last_key) = number;
           idx_last_key = idx_last_key + 1;
           continue;
        end
        %if this is the response value
        if strcmp('#label#', keyword)
            Y_all(i) = number;
            continue;
        end
        idx = find(strcmp(keywords_all, keyword));
        if (idx > 0)
            X_all_raw(i,idx) = number;
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

save('amazon_data_raw','X_all_raw','Y_all','keywords_all','-v7.3');

