SUMMARY
==============================================
This package is the code for project 3 of EE239AS Special Topics in Signals and Systems. The author are Liqiang YU, Kaiming WANG and Jun FENG. The program are based on MATLAB R2015b.

BEFORE USEAGE
==============================================
Before using the code, please ensure to include wnmfrule function in the Matrix Factorization Toolbox. 

DETAILED DESCRIPTIONS OF FILES
==============================================
Please include all the files in your working space.

Please run problem*.m respectively.

Here are brief descriptions of the files.

data.mat	—-The data of u.data from 100k MovieLens data sets. More information, 		please visit http://www.movielens.org/

squareError.m  —-calculate square error between R and U*V

precisionAndRecall.m	—-calculate precision and recall between actual data and predicted data, threshold of actual data is always 3 while threshold of predicted data can be changed

roc_grapher.m  —-load data from .mat and draw the precision VS recall curve. Also 			calculates the area under it.
myGetPrecision.m ——calculate precision for problem 5.
getHit_FalseAlarm.m ——calculate hit rate and false alarm rate for problem 5.

wnmf_new_reg.m — the function that solve the weighted non-negative matrix 				factorization with regularization.

=========================================================================
problem1.m	—-Please run directly. This is the code for problem1.
problem2.m	—-Please run directly. This is the code for problem2.
problem3.m	—-Please run directly. This is the code for problem3.
problem4_part1.m —-Please run directly. This is the code for the first part of 			problem4.
problem4_new_part2.m —-Please run directly. This is the code for the second part of 		problem4.
problem5_1.m	——Please run directly. This is the code for problem5 to get the 			precision.
problem5_2.m	——You need to run problem5_1.m first and then run this program. This 		is the code for problem 5 to get the hit rate and false alarm rate.
