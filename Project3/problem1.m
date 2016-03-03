clear;
load data;
[A1,Y1]=wnmfrule(data,10);
error1=squareError(data,A1,Y1);
[A2,Y2]=wnmfrule(data,50);
error2=squareError(data,A2,Y2);
[A3,Y3]=wnmfrule(data,100);
error3=squareError(data,A3,Y3);
disp(['when k is 10,  the total least squared error is ' num2str(error1)])
disp(['when k is 50,  the total least squared error is ' num2str(error2)])
disp(['when k is 100,  the total least squared error is ' num2str(error3)])