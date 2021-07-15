# Weakly-Supervised Classification and Detection of Bird Sounds in the Wild. A BirdCLEF 2021 Solution.

## 10th Place Solution to BirdCLEF 2021 - Birdcall Identification 
Repository contains the code for 10th place solution of [BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021/leaderboard) Competition. When submitted to CLEF 2021 Conference, the [paper](https://arxiv.org/abs/2107.04878) was awarded best working note award. 

## Task 
Given a long audio in .𝑜𝑔𝑔 format, the goal of the contest was to predict if there is a bird call in each 5-seconds segment of the given soundscape, and identify which of the 397 birds is in such segment. Models had to infer on the test set within 3 hours run-time limit to ensure the efficiency of the solutions.

## Approach
We based our solution on diverse and robust models trained on a complete audio dataset using custom augmentations, and on a postprocess algorithm that improves the predicted probabilities of bird appearances by using additional features as the site (longitude, latitude), rarity of the bird, appearance of other birds in the audio, etc.
![approach](https://github.com/kumar-shubham-ml/kaggle-birdclef-2021/blob/main/data/approach.png)
We bagged 13 CNN-based models, which were different in terms of augmentation strategy and architecture. We used a SVC model for probability calibration, and a haversine distance based post processing for Geofencing and for creating bird to site mapping to reduce False Positives and False Negatives.

## Results

| Method | All Sites (2021) || COR Site ||| SSW Site ||| COR & SSW Sites |||
|| Public LB | Private LB | No call | Call | CV@0.54 | No call | Call | CV@0.54 | No call | Call | CV@0.54 |
| :------: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SNE & SSW site models | - | - | - | - | - | 0.9094 | 0.5552 | 0.7465 | - | - | - |
| All site models | 0.7155 | 0.6203 | 0.9300 | 0.5208 | 0.7418 | 0.9431 | 0.3876 | 0.6875 | 0.9261 | 0.4623 | 0.7127 |
| Ensemble | 0.7499 | 0.6450 | 0.9300 | 0.5208 | 0.7418 | 0.8923 | 0.5861 | 0.7514 | 0.9130 | 0.5591 | 0.7502 |
| Ensemble + PC | 0.7744 | 0.6609 | 0.9187 | 0.6415 | 0.7912 | 0.8869 | 0.6106 | 0.7598 | 0.9044 | 0.6234 | 0.7751 |
| Ensemble + PC + Site-info | 0.7711 | 0.6722 | 0.9106 | 0.6756 | 0.8025 | 0.8725 | 0.6327 | 0.7622 | 0.8934 | 0.6505 | 0.7816 |
| Ensemble + PC + FNR | 0.7774 | 0.6630 | 0.9086 | 0.6758 | 0.8015 | 0.8720 | 0.6354 | 0.7632 | 0.8921 | 0.6521 | 0.7817 |
| Ensemble + PC + FNR + FPR | 0.7754  | 0.6780 | 0.9285 | 0.6583 | 0.8029 | 0.8836 | 0.6343 | 0.7656 | 0.9082 | 0.6443 | 0.7836 |
| Selected Submission | 0.7801 | 0.6738 | 0.9106 | 0.6756 | 0.8025 | 0.8754 | 0.6363 | 0.7654 | 0.8947 | 0.6526 | 0.7834 |

## Citation
