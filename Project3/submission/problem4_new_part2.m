load('data.mat');
%load('problem2.mat')


obervation=[userId,itemId,rating];
% error=zeros(1,10);
indices = crossvalind('Kfold',100000,10);
%allabs=zeros(1,10);

lambdas=[0.01,0.1,1];
k=10;
crossValNum=1;

As=zeros(length(lambdas),crossValNum,943,k);
Ys=zeros(length(lambdas),crossValNum,k,1682);
finalResiduals=zeros(length(lambdas),crossValNum);
tic;
for lambda_index=1:length(lambdas)
    for j = 1:1:crossValNum
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
        option.iter=200;
        %     option.dis=false;
        %k=50;
        lambda=lambdas(lambda_index);
        [A,Y,numIter,tElapsed,finalResidual]=wnmf_new_reg(R,k,lambda,option);
        As(lambda_index,j,:,:)=A;
        Ys(lambda_index,j,:,:)=Y;
        finalResiduals(lambda_index,j)=finalResidual;
        P=A*Y;
        testset=obervation(test,:);
        currentabs=0;
        targets=(testset(:,3)>3);
        scores=zeros(length(targets),1);
        for n=1:1:length(testset(:,1))
            curuser=testset(n,1);
            curitem=testset(n,2);
            %currating=testset(n,3);
            scores(n)=P(curuser,curitem);
        end
        predict_1=sum((scores>3));
        predict_0=length(targets)-predict_1;
        real_1=sum(targets);
        real_0=length(targets)-real_1;
        tp=sum((scores>3) & targets);
        precision=tp/predict_1;
        recall=tp/real_1;
        thresholds=-1:0.001:3;
        precisions=zeros(length(thresholds),1);
        %     precisions(length(thresholds)+2)=1;
        recalls=zeros(length(thresholds),1);
        %     recalls(1)=1;
        for i=1:length(thresholds)
            precisions(i)= sum((scores>thresholds(i)) & targets)/sum((scores>thresholds(i)));
            recalls(i)=sum((scores>thresholds(i)) & targets)/sum(targets);
        end
        precisions(isnan(precisions))=0;
        AUC=sum(precisions(1:length(thresholds)-1).*-(recalls(2:length(thresholds))-recalls(1:length(thresholds)-1)));
        figure(1)
        %[Xpr,Ypr,Tpr,AUCpr] = perfcurve(targets, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
        plot([recalls],[precisions])
        hold on
        xlabel('Recall'); ylabel('Precision')
        legend(['lambda=',num2str(lambdas(1))],['lambda=',num2str(lambdas(2))],['lambda=',num2str(lambdas(3))])
        title(['Precision-recall curve'])
    end
end;
toc
save('problem4_new_part2_k10_iter200_03042212');
% error=sum(allabs)/10;
% save('p4_part2');
% error
% precision,recall,AUC