clear;
load data;
obervation=[userId,itemId,rating];
error=zeros(1,10);
indices = crossvalind('Kfold',100000,10);
result=zeros(2,10);
option.iter=1000;
option.dis=false;
for j = 1:1:10
    test = (indices == j); 
    train = ~test;
    R=NaN*ones(943,1682);
    trainset=obervation(train,:);
    for m=1:1:90000
         curuser=trainset(m,1);
         curitem=trainset(m,2);
         currating=trainset(m,3);
         R(curuser,curitem)=currating;
    end
    [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100,option);
    P=A*Y;
    testset=obervation(test,:);
    prerating=NaN*ones(1,10000);
    for i=1:1:10000
        tmpuserId=testset(i,1);
        tmpitemId=testset(i,2);
        prerating(i)=P(tmpuserId,tmpitemId);
    end
    [precision,recall]=precisionAndRecall(testset(:,3),prerating,3);
    result(1,j)=precision;
    result(2,j)=recall;
end
precisions=zeros(1,1001);
recalls=zeros(1,1001);
for i=0:1:1000
    [precision,recall]=precisionAndRecall(testset(:,3),prerating,i*0.01);
    precisions(i+1)=precision;
    recalls(i+1)=recall;
end;
figure
plot(recalls,precisions)
xlabel('Recall'); ylabel('Precision')
AUC=-trapz(recalls,precisions);
title(['Precision-recall curve (AUC=' num2str(AUC) ')'])