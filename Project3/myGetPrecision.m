function p = myGetPrecision(predict_index, data, usr, varargin)
    count = 0;
    for i = 1:size(predict_index,2)
        if(data(usr, predict_index(i)) > 3)
            count = count+1;
        end
    end
    p = count / size(predict_index,2);
end