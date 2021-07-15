# Weakly-Supervised Classification and Detection of Bird Sounds in the Wild. A BirdCLEF 2021 Solution.

## 10th Place Solution to BirdCLEF 2021 - Birdcall Identification 
Repository contains the code for 10th place solution of [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021/leaderboard) Competition. When submitted to CLEF 2021 Conference, the [paper](https://arxiv.org/abs/2107.04878) was awarded best working note award. 

## About
Given a long audio in .𝑜𝑔𝑔 format, the goal of the contest was to predict if there is a bird call in each 5-seconds segment of the given soundscape, and identify which of the 397 birds is in such segment. Models inferred on the test set within 3 hours run-time limit to ensure the efficiency of the solutions.

## Approach
We base our solution on diverse and robust models trained on a complete audio dataset using custom augmentations, and on a postprocess algorithm that improves the predicted probabilities of bird appearances by using additional features as the site (longitude, latitude), rarity of the bird, appearance of other birds in the audio, etc.
![approach](https://github.com/kumar-shubham-ml/kaggle-birdclef-2021/blob/main/data/approach.png)

## Results


## Citation
