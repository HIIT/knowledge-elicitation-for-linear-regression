%% removes non alphanumeric characters from lowercase string

function  output_word = alphanumeric(input_word)

    output_word=''; 
    j=1;
    
    for i=1:length(input_word)
        if (input_word(i) >= 'a' && input_word(i) <='z')
            output_word(1,j)= input_word(i);
            j=j+1;
        end
    end
    
end
