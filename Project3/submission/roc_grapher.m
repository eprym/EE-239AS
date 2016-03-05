load('problem4_new_part2_k100_iter200_03042112.mat');
load('data.mat')
% A=zeros(943,k);
% Y=zeros(k,1682);
for lambda_index=1:length(lambdas)
    for j = 1:1:crossValNum
        test = (indices == j);
        
        A=reshape(As(lambda_index,j,:,:),943,k);
        Y=reshape(Ys(lambda_index,j,:,:),k,1682);
        finalResidual=finalResiduals(lambda_index,j);
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
        thresholds=-1:0.01:5;
        precisions=zeros(length(thresholds),1);
        %     precisions(length(thresholds)+2)=1;
        recalls=zeros(length(thresholds),1);
        %     recalls(1)=1;
        for i=1:length(thresholds)
            precisions(i)= sum((scores>thresholds(i)) & targets)/sum((scores>thresholds(i)));
            recalls(i)=sum((scores>thresholds(i)) & targets)/sum(targets);
%             [precisions(i),recalls(i)]=precisionAndRecall(targets,scores,thresholds(i));
        end
        precisions(isnan(precisions))=0;
        AUC=sum(precisions(1:length(thresholds)-1).*-(recalls(2:length(thresholds))-recalls(1:length(thresholds)-1)));
        figure(1)
        %[Xpr,Ypr,Tpr,AUCpr] = perfcurve(targets, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
        plot([recalls],[precisions])
        hold on;
        xlabel('Recall'); ylabel('Precision')
        axis([0 1 0 1]);
%         legend(['lambda=',num2str(lambdas(lambda_index)),',k=',num2str(k)]);
        title(['Precision-recall curve (AUC: ' num2str(AUC) ')'])
    end
end;