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

<table>
    <tr>
        <th> Method </th> <th colspan = "2" style="text-align:center"> All Sites (2021) </th> <th colspan = "3" style="text-align:center"> COR Site </th> <th colspan = "3" style="text-align:center"> SSW Site </th> <th colspan = "3" style="text-align:center"> COR & SSW Sites </th> </tr> <tr>
<td> </td> <td style="text-align:center"> Public LB </td> <td style="text-align:center"> Private LB </td> <td style="text-align:center"> No call </td> <td style="text-align:center"> Call </td> <td style="text-align:center"> CV@0.54 </td> <td style="text-align:center"> No call </td> <td style="text-align:center"> Call </td> <td style="text-align:center"> CV@0.54 </td> <td style="text-align:center"> No call </td> <td style="text-align:center"> Call </td> <td style="text-align:center"> CV@0.54 </td> </tr> <tr>
<td> SNE & SSW site models </td> <td> - </td> <td> - </td> <td> - </td> <td> - </td> <td> - </td> <td> 0.9094 </td> <td> 0.5552 </td> <td> 0.7465 </td> <td> - </td> <td> - </td> <td> - </td> </tr> <tr>
<td> All site models </td> <td> 0.7155 </td> <td> 0.6203 </td> <td> 0.9300 </td> <td> 0.5208 </td> <td> 0.7418 </td> <td> 0.9431 </td> <td> 0.3876 </td> <td> 0.6875 </td> <td> 0.9261 </td> <td> 0.4623 </td> <td> 0.7127 </td> </tr> <tr>
<td> Ensemble </td> <td> 0.7499 </td> <td> 0.6450 </td> <td> 0.9300 </td> <td> 0.5208 </td> <td> 0.7418 </td> <td> 0.8923 </td> <td> 0.5861 </td> <td> 0.7514 </td> <td> 0.9130 </td> <td> 0.5591 </td> <td> 0.7502 </td> </tr> <tr>
<td> Ensemble + PC </td> <td> 0.7744 </td> <td> 0.6609 </td> <td> 0.9187 </td> <td> 0.6415 </td> <td> 0.7912 </td> <td> 0.8869 </td> <td> 0.6106 </td> <td> 0.7598 </td> <td> 0.9044 </td> <td> 0.6234 </td> <td> 0.7751 </td> </tr> <tr>
<td> Ensemble + PC + Site-info </td> <td> 0.7711 </td> <td> 0.6722 </td> <td> 0.9106 </td> <td> 0.6756 </td> <td> 0.8025 </td> <td> 0.8725 </td> <td> 0.6327 </td> <td> 0.7622 </td> <td> 0.8934 </td> <td> 0.6505 </td> <td> 0.7816 </td> </tr> <tr>
<td> Ensemble + PC + FNR </td> <td> 0.7774 </td> <td> 0.6630 </td> <td> 0.9086 </td> <td> 0.6758 </td> <td> 0.8015 </td> <td> 0.8720 </td> <td> 0.6354 </td> <td> 0.7632 </td> <td> 0.8921 </td> <td> 0.6521 </td> <td> 0.7817 </td> </tr> <tr>
<td> Ensemble + PC + FNR + FPR </td> <td> 0.7754  </td> <td> 0.6780 </td> <td> 0.9285 </td> <td> 0.6583 </td> <td> 0.8029 </td> <td> 0.8836 </td> <td> 0.6343 </td> <td> 0.7656 </td> <td> 0.9082 </td> <td> 0.6443 </td> <td> 0.7836 </td> </tr> <tr>
<td> Selected Submission </td> <td> 0.7801 </td> <td> 0.6738 </td> <td> 0.9106 </td> <td> 0.6756 </td> <td> 0.8025 </td> <td> 0.8754 </td> <td> 0.6363 </td> <td> 0.7654 </td> <td> 0.8947 </td> <td> 0.6526 </td> <td> 0.7834 </td> </tr> <tr>
</table>


## Citation
