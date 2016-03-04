function p = myGetPrecision2(predict_index, data_index)
p = size(intersect(predict_index, data_index),2) / size(predict_index,2);
end