function[precision,recall]=precisionAndRecall(obseration,prediction,threshold)
    x=obseration>3;
    y=prediction>threshold;
    l=length(obseration);
    count1=0;
    count2=0;
    count3=0;
    count4=0;
    for i=1:1:l
        if x(i)==1
            count1=count1+1;
            if y(i)==1
                count2=count2+1;
            end
        end
        if y(i)==1
            count3=count3+1;
            if x(i)==1
                count4=count4+1;
            end
        end
    end
    precision=count2/count1;
    recall=count4/count3;    
end