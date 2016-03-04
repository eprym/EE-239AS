clear;
close all;
clc;

load('/data.mat')
obervation=[userId,itemId,rating];

data(isnan(data)) = 0;
option.dis = false;
option.iter = 100;
k = 10;
indices = crossvalind('Kfold',100000,k);
allabs=zeros(1,k);
totalPrecision = 0;
parfor j = 1:1:k
    test = (indices == j); 
    train = ~test;
    R=NaN*ones(943,1682);
    trainset=obervation(train,:); 
    for m=1:1:100000*(k-1)/k
        curuser=trainset(m,1);
        curitem=trainset(m,2);
        currating=trainset(m,3);
        R(curuser,curitem)=currating;
    end
    W = ones(size(R));
    W(isnan(R)) = 0;
    R(isnan(R)) = 0;
    %[A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100,option);
    [A,Y,numIter,tElapsed,finalResidual]=wnmf_reg(R,100,0.1,option);
    P=A*Y*5;
    testset=obervation(test,:);
    currentabs=0;
    R_test = NaN * ones(943, 1682);
    for n=1:1:100000*1/k
        curuser=testset(n,1);
        curitem=testset(n,2);
        currating=testset(n,3);
        R_test(curuser,curitem)=currating;
        currentabs=currentabs+abs(P(curuser,curitem)-currating);
    end
    allabs(j)=currentabs/(100000*1/k);
    R_test(isnan(R_test)) = 0;
    
    precision = [];
    L = 5;
%     for p = 1:size(P,1)
%         [predict_result, predict_index] = sort(P(p,:), 'descend');
%         [data_result, data_index] = sort(R_test(p,:), 'descend');
%         precision = [precision, myGetPrecision(data_index(1:L),P, p)];
%         %precision = [precision, myGetPrecision2(predict_index(1:L),data_index(1:L))];
%     end
    for p = 1:size(P,1)
        P_filter = P(p, ismember(P(p,:), R_test(p,:)));
        [predict_result, predict_index] = sort(P_filter, 'descend');
        if(size(predict_index,2)>=L)
            precision = [precision, myGetPrecision(predict_index(1:L),R_test, p)];
        elseif(size(predict_index,2) ~= 0)
            precision = [precision, myGetPrecision(predict_index,R_test, p)];
        end
    end
    fprintf('The average precision for test %d is %f\n', j, mean(precision));
    totalPrecision = totalPrecision + mean(precision); 
end

%save('P.mat', 'P')

fprintf('The total average precision is %f\n', totalPrecision/k);




