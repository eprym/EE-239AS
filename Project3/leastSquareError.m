function error=leastSquareError(R,U,V)
    error=0;
    s=size(R);
    n=isnan(R);
    P=U*V;
    for i=1:1:s(1)
        for j=1:1:s(2)
            if n(i,j)==0
                error=error+(R(i,j)-P(i,j))^2;
            end
        end
    end
    

end