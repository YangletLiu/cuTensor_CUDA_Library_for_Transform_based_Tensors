Uc = zeros(size(U))
Sc = zeros(size(S))
Vc = zeros(size(V))
for i=1:100
    Uc(:,i,:) = U(:,i,:)
    Sc(i,i,:) = S(i,i,:)
    Vc(:,i,:) = V(:,i,:)
end
USc = tprod(Uc,Sc)
USVc = tprod(Sc,Vc)
