load('data.mat');
load('problem2.mat');

aobervation=[userId,itemId,rating];
% error=zeros(1,10);
indices = crossvalind('Kfold',100000,10);
allabs=zeros(1,10);
tic;
for j = 1:1:1
    test = (indices == j);
    train = ~test;
    R=NaN*ones(943,1682);
    trainset=obervation(train,:);
    for m=1:1:length(trainset(:,1))
        curuser=trainset(m,1);
        curitem=trainset(m,2);
        currating=trainset(m,3);
        R(curuser,curitem)=currating;
    end
    option.iter=1000;
%     option.dis=false;
    k=10;
    lambda=0.1;
    [A,Y,numIter,tElapsed,finalResidual]=wnmf_reg(R,k,lambda,option);
    P=A*Y;
    testset=obervation(test,:);
    currentabs=0;
    targets=(testset(:,3)>3);
    scores=zeros(length(targets),1);
    for n=1:1:length(testset(:,1))
        curuser=testset(n,1);
        curitem=testset(n,2);
        currating=testset(n,3);
        scores(n)=currating*P(curuser,curitem);
    end
    predict_1=sum((scores>3));
    predict_0=length(targets)-predict_1;
    real_1=sum(targets);
    real_0=length(targets)-real_1;
    tp=sum((scores>3) & targets);
    precision=tp/predict_1;
    recall=tp/real_1;
    thresholds=0:0.01:10;
    precisions=zeros(length(thresholds),1);
%     precisions(length(thresholds)+2)=1;
    recalls=zeros(length(thresholds),1);
%     recalls(1)=1;
    for i=1:length(thresholds)
       precisions(i)= sum((scores>thresholds(i)) & targets)/sum((scores>thresholds(i)));
       recalls(i)=sum((scores>thresholds(i)) & targets)/sum(targets);
    end
    precisions(isnan(precisions))=1;
    AUC=sum(precisions(1:length(thresholds)-1).*-(recalls(2:length(thresholds))-recalls(1:length(thresholds)-1)));
    figure(1)
    %[Xpr,Ypr,Tpr,AUCpr] = perfcurve(targets, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
    plot([recalls],[precisions])
    xlabel('Recall'); ylabel('Precision')
    title(['Precision-recall curve (AUC: ' num2str(AUC) ')'])
end
toc
% error=sum(allabs)/10;
% save('p4_part2');
% error
precision,recall,AUC