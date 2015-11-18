function [NoisyFeatures]=AddNoise(Features,DecimalNoiseToAdd)

NoiseToAdd=randn(size(Features));

NoisyFeatures=Features+NoiseToAdd.*Features*DecimalNoiseToAdd;
