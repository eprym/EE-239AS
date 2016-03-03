function p = getPrecision(predict_index, test_index, L, data, usr)
    count = 0;
    for i = 1:L
        for j = 1:L
            if(predict_index(i) == test_index(j) && data(usr, test_index(j)) ~= -1)
                count = count+1;
                break;
            end
        end
    end
    p = count / L;
end