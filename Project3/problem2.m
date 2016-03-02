obervation=[userId,itemId,rating];
error=zeros(1,10);
for i=1:1:10
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
        [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,100);
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
    error(i)=sum(allabs)/10;
end