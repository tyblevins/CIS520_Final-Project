function PredictedLabel=KNearestNeighbor(Features,Labels,FeaturesTest,k)

ClosestMatches = knnsearch(Features,FeaturesTest,'K',k);

for i=1:k
LabelOfClosestMatch(:,i)=Labels(ClosestMatches(:,i));
end
PredictedLabel=round(mean(LabelOfClosestMatch,2));
RandomGuesses=round(rand(sum(mean(LabelOfClosestMatch,2)==.5),1));
PredictedLabel(mean(LabelOfClosestMatch,2)==.5)=RandomGuesses;