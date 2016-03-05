clear;
load data;
obervation=[userId,itemId,rating];
iteration=[50,100,200];
AUC=zeros(1,3);
figure(1)
hold on
title('Precision-recall Curve for k=100 under Different Iterations')
xlabel('Recall') 
ylabel('Precision')
for j=1:1:3
    error=zeros(1,10);
    indices = crossvalind('Kfold',100000,10);
    result=zeros(2,10);
    option.iter=iteration(j);
    option.dis=false;

    test = (indices == 1); 
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

    precisions=zeros(1,1001);
    recalls=zeros(1,1001);
    for i=0:1:1000
        [precision,recall]=precisionAndRecall(testset(:,3),prerating,i*0.01);
        precisions(i+1)=precision;
        recalls(i+1)=recall;
    end;


    plot(recalls,precisions)
    AUC(j)=-trapz(recalls,precisions);
end
legend(['iteration= ' num2str(iteration(1)) ',AUC= ' num2str(AUC(1))],['iteration= ' num2str(iteration(2)) ',AUC= ' num2str(AUC(2))],['iteration= ' num2str(iteration(3)) ',AUC= ' num2str(AUC(3))])

figure(2)
hold on
iteration=[10,50,100];
AUC=zeros(1,3);
title('Precision-recall Curve for iteration=100 under Different k')
xlabel('Recall') 
ylabel('Precision')
for j=1:1:3
    error=zeros(1,10);
    indices = crossvalind('Kfold',100000,10);
    result=zeros(2,10);
    option.iter=100;
    option.dis=false;

    test = (indices == 1); 
    train = ~test;
    R=NaN*ones(943,1682);
    trainset=obervation(train,:);
    for m=1:1:90000
         curuser=trainset(m,1);
         curitem=trainset(m,2);
         currating=trainset(m,3);
         R(curuser,curitem)=currating;
    end

    [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,iteration(j),option);
    P=A*Y;
    testset=obervation(test,:);
    prerating=NaN*ones(1,10000);

    for i=1:1:10000
        tmpuserId=testset(i,1);
        tmpitemId=testset(i,2);
        prerating(i)=P(tmpuserId,tmpitemId);
    end

    precisions=zeros(1,1001);
    recalls=zeros(1,1001);
    for i=0:1:1000
        [precision,recall]=precisionAndRecall(testset(:,3),prerating,i*0.01);
        precisions(i+1)=precision;
        recalls(i+1)=recall;
    end;


    plot(recalls,precisions)
    AUC(j)=-trapz(recalls,precisions);
end
legend(['k= ' num2str(iteration(1)) ',AUC= ' num2str(AUC(1))],['k= ' num2str(iteration(2)) ',AUC= ' num2str(AUC(2))],['k= ' num2str(iteration(3)) ',AUC= ' num2str(AUC(3))])
