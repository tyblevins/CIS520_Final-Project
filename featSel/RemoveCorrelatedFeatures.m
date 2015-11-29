function [EditedFeatures,FeaturesToInclude]=RemoveCorrelatedFeatures(Features,MaximumCorrelation)
FeaturesToInclude=1:size(Features,2);
Correlation=corr(Features(:,FeaturesToInclude))-eye(length(FeaturesToInclude));
while max(max(Correlation(FeaturesToInclude,FeaturesToInclude)))>MaximumCorrelation
    [~,IndexOfMax]=sort(max(Correlation(FeaturesToInclude,FeaturesToInclude)),'descend');
    FeaturesToInclude(IndexOfMax(1))=[];
end

EditedFeatures=Features(:,FeaturesToInclude);