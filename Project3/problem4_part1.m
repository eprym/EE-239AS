load('data.mat');
load('problem2.mat');

obervation=[userId,itemId,rating];
% error=zeros(1,10);
indices = crossvalind('Kfold',100000,10);
allabs=zeros(1,10);
tic;
R=NaN*ones(943,1682);
trainset=obervation;
[m,n]=size(trainset);
K=[10,50,100];
lambdas=[0.01,0.1,1];
K_len=length(K);
lambdas_len=length(lambdas);
finalResiduals=zeros(2,K_len,lambdas_len);
for K_index=1:K_len
    for lambda_index=1:lambdas_len
        for i=1:1:m;
            curuser=trainset(i,1);
            curitem=trainset(i,2);
            currating=trainset(i,3);
            R(curuser,curitem)=currating;
        end
        option.iter=1000;
        option.dis=false;
        k=K(K_index);
        lambda=lambdas(lambda_index);
        disp(['k=',num2str(k),';   lambda=',num2str(lambda)]);
        [A,Y,numIter,tElapsed,finalResidual]=wnmf_reg(R,k,lambda,option);
        finalResiduals(1,K_index,lambda_index)=finalResidual;
        [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(R,k,option);
        finalResiduals(2,K_index,lambda_index)=finalResidual;
    end;
end;
% P=A*Y;
% testset=obervation;
% sqr_err=0;
% for i=1:1:m
%     curuser=testset(i,1);
%     curitem=testset(i,2);
%     currating=testset(i,3);
%     sqr_err=sqr_err+(P(curuser,curitem)-currating)^2;
% end
% allabs(j)=sqr_err;
toc
% error=sum(allabs)/10;
% save('p4_part1');
finalResiduals