clear;
close all;
clc;

load('/data.mat')
obervation=[userId,itemId,rating];

data(isnan(data)) = 0;
option.dis = true;
option.iter = 500;
k = 5;
indices = crossvalind('Kfold',100000,k);
allabs=zeros(1,k);
totalPrecision = 0;
for j = 1:1:k
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
    R(isnan(R)) = 0;
    [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100,option);
    %[A,Y,tElapsed]=wnmf_reg(W,R,100,0.1,option);
    P=A*Y;
    testset=obervation(test,:);
    currentabs=0;
    for n=1:1:100000*1/k
        curuser=testset(n,1);
        curitem=testset(n,2);
        currating=testset(n,3);
        currentabs=currentabs+abs(P(curuser,curitem)-currating);
    end
    allabs(j)=currentabs/(100000*1/k);
    
    precision = [];
    L = 5;
    for p = 1:size(P,1)
        [predict_result, predict_index] = sort(P(p,:), 'descend');
        precision = [precision, myGetPrecision(predict_index(1:L),data, p)];
    end
    fprintf('The average precision for test %d is %f\n', j, mean(precision));
    totalPrecision = totalPrecision + mean(precision); 
end

save('P.mat', P)

fprintf('The total average precision is %f\n', totalPrecision/k);




