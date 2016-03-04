function [totalHitRate, totalFalseAlarm] = get_Hit_FalseAlarm(P,data)
% clear;
% clc;
% load('P.mat');
% load('data.mat');

L_end = 20;

[P_sorted, index] = sort(P,2,'descend');
totalHitRate = zeros(1, L_end);
totalFalseAlarm = zeros(1, L_end);
parfor L = 1:L_end
    fprintf('calculate hit and false alarm rate for top %d movies...\n', L)
    recommendID = index(:,1:L);
%     for i=1:size(P,1)
%         recommendSet = union(recommendSet, recommendID(i,:));
%     end
    
    hitrate = 0;
    falsealarm = 0;
    for i=1:size(P,1)
        likeID = find(data(i,:) > 3);
        dislikeID = intersect(find(data(i,:)<4), find(data(i,:)>0));
        unknownID = find(data(i,:) == 0);
        recommend_filter = recommendID(i,~ismember(recommendID(i,:), unknownID));
        if(size(likeID,2) ~= 0)
            hitrate = hitrate + size(intersect(likeID, recommend_filter),2) / size(likeID,2);
        end
        if(size(dislikeID,2) ~= 0)
            falsealarm = falsealarm + size(intersect(dislikeID, recommend_filter),2) / size(dislikeID,2);
        end
        
    end
    totalHitRate(L) = hitrate / size(P,1);
    totalFalseAlarm(L) = falsealarm / size(P,1);
end
figure;
plot(totalFalseAlarm, totalHitRate);
