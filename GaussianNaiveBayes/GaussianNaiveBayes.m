function PredictedLabel=GaussianNaiveBayes(Features,Labels,FeaturesTest)

%% Remove features with 0 variance
FeaturesWithVariance=var(Features)~=0 & var(FeaturesTest)~=0;
Features=Features(:,FeaturesWithVariance);
FeaturesTest=FeaturesTest(:,FeaturesWithVariance);
for Outcome=1:length(unique(Labels))
        ProbabilityOfLabel(Outcome)=sum(Labels==Outcome-1)/numel(Labels); 
        FeaturesOfLabel=Features(Labels==Outcome-1,:);
        MeanFeaturesOfLabel(Outcome,:)=mean(FeaturesOfLabel); 
        STDFeaturesOfLabel(Outcome,:)=std(FeaturesOfLabel); 
end

for Outcome=1:size(MeanFeaturesOfLabel,1);
    for AFeature=1:size(FeaturesTest,2)
        ProbabilityOfFeatureGivenLabel(:,Outcome,AFeature)=normpdf(FeaturesTest(:,AFeature),MeanFeaturesOfLabel(Outcome,AFeature),STDFeaturesOfLabel(Outcome,AFeature));
    end
end
    ProbabilityOfAllFeaturesGivenLabel=prod(ProbabilityOfFeatureGivenLabel,3);
    ProbabilityOfLetterGivenAllFeatures=(ProbabilityOfLabel'*ones(1,size(ProbabilityOfAllFeaturesGivenLabel,1)))'.*ProbabilityOfAllFeaturesGivenLabel;
    
    [~,PredictedLabel]=max(ProbabilityOfLetterGivenAllFeatures,[],2);
    PredictedLabel=PredictedLabel-1;
end