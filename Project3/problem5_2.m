clear;
clc;
load('P.mat');
load('data.mat');
data(isnan(data)) = 0;
[hit, false_alarm] = get_Hit_FalseAlarm(P,data);