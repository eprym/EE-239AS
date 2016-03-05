clear;
load data;
obervation=[userId,itemId,rating];
error=zeros(1,10);
indices = crossvalind('Kfold',100000,10);
allabs=zeros(1,10);
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
    option.iter=1000;
    [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100,option);
    P=A*Y;
    testset=obervation(test,:);
    currentabs=0;
    for n=1:1:10000
        curuser=testset(n,1);
        curitem=testset(n,2);
        currating=testset(n,3);
        currentabs=currentabs+abs(P(curuser,curitem)-currating);
    end
    allabs(j)=currentabs;
end
allabs=allabs/10000;
disp(['average absolute error over testing data for each entry of all 10 tests is ' num2str(sum(allabs)/10)])
disp(['highest average absolute error over testing data for each entry is ' num2str(max(allabs))])
disp(['lowest average absolute error over testing data for each entry is ' num2str(min(allabs))])