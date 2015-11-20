function PValue=TTestForFeatures(Features,Labels)
PValue=zeros(size(Features,2),1);
for i=1:size(Features,2)
    [~,PValue(i)]=ttest2(Features(Labels==0,i),Features(Labels==1,i));
end