obervation=[userId,itemId,rating];
error=zeros(1,10);
indices = crossvalind('Kfold',100000,10);
result=zeros(2,10);
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
    [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100);
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
precisions=zeros(1,501);
recalls=zeros(1,501);
for i=0:1:500
    [precision,recall]=precisionAndRecall(testset(:,3),prerating,i*0.01);
    precisions(i+1)=precision;
    recalls(i+1)=recall;
end;
plot(recalls,precisions)
xlabel('Recall'); ylabel('Precision')
title(['Precision-recall curve (AUC: ' trapz(precisions(1:500), -1./recalls(1:500)) ')'])