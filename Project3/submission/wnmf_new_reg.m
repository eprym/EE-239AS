function [A,Y,numIter,tElapsed,finalResidual]=wnmf_new_reg(X,k,lambda,option)
% Weighted NMF based on multiple update rules for missing values: X=AY, s.t. A,Y>=0.
% Definition:
%     [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(X,k)
%     [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(X,k,option)
% X: non-negative matrix, dataset to factorize, each column is a sample,
% and each row is a feature. A missing value is represented by NaN.
% k: number of clusters.
% option: struct:
% option.distance: distance used in the objective function. It could be
%    'ls': the Euclidean distance (defalut),
%    'kl': KL divergence.
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate.
%    If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the basis matrix.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
%%%%


tStart=tic;
optionDefault.distance='ls';
optionDefault.iter=100;
optionDefault.dis=true;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
    option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% Weight
W=isnan(X);
X(W)=0;
W=~W;

% iter: number of iterations
[r,c]=size(X); % c is # of samples, r is # of features
Y=rand(k,c);
% Y(Y<eps)=0;
Y=max(Y,eps);
A=W/Y;
% A(A<eps)=0;
A=max(A,eps);
XfitPrevious=Inf;
for n=1:option.iter
    YY = Y*Y';
    parfor i=1:r
        Ci=diag(X(i,:));
        %A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        A(i,:)=((YY+Y*(Ci-eye(c))*Y'+lambda*eye(k))^(-1)*Y*Ci*W(i,:)')';
        %             A(A<eps)=0;
    end
    A=max(A,eps);
    %Y=pinv(A)*W;
%     AA=A'*A;
%     parfor j=1:c
%         Cj=diag(X(:,j));
%         %Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
%         Y(:,j)=((AA+A'*(Cj-eye(r))*A+lambda*eye(k))^(-1)*A'*Cj*(W(:,j)));
%         %             Y(Y<eps)=0;
%     end
    Y=pinv(A)*W;
    Y=max(Y,eps);
    
    if mod(n,10)==0 || n==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(n),'th']);
        end
        XfitThis=A*Y;
        fitRes=matrixNorm(W.*(XfitPrevious-XfitThis));
        XfitPrevious=XfitThis;
        curRes=norm(W.*(X-XfitThis),'fro');
        if option.tof>=fitRes || option.residual>=curRes || n==option.iter
            s=sprintf('Mutiple update rules based NMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',n,curRes);
            disp(s);
            numIter=n;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
