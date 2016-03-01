for i=1:1:100000
    item=itemId(i);
    user=userId(i);
    data(user,item)=rating(i);
end