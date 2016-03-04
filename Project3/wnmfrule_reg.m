function [A,Y,numIter,tElapsed,finalResidual]=wnmfrule_reg(X,k,lambda,option)
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
% Copyright (C) <2012>  <Yifeng Li>
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 01, 2011
%%%%


tStart=tic;
optionDefault.distance='ls';
optionDefault.iter=1000;
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
WfitPrevious=Inf;
for n=1:option.iter
    A=A.*(((X.*W)*Y')./(((X.*(A*Y))*Y')+lambda*A));
%     for i=1:r
%         Ci=diag(X(i,:));
%         %A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
%         A(i,:)=((Y*Ci*Y'+lambda*eye(k))^(-1)*Y*Ci*(W(i,:))')';
%         %             A(A<eps)=0;
%     end
    A=max(A,eps);
%     Y=pinv(A)*W;
%     for j=1:c
%         Cj=diag(r(:,j));
%         %Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
%         Y(:,j)=((X'*Cj*X+lambda*eye(k))^(-1)*X'*Cj*(W(:,j)));
%         %             Y(Y<eps)=0;
%     end
    Y=Y.*((A'*(X.*W))./(A'*(X.*(A*Y))+lambda.*Y));
    Y=max(Y,eps);
    
    if mod(n,10)==0 || n==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(n),'th']);
        end
        WfitThis=A*Y;%WX
        fitRes=matrixNorm(sqrt(X).*(WfitPrevious-WfitThis));%WX
        WfitPrevious=WfitThis;
        curRes=norm(sqrt(X).*(W-WfitThis),'fro');%WX
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
