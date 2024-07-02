# Diversity-based Clustering and Shapley-based GRS
Codes related to a project focusing on group recommendation.

### project Description
This project presents an innovative approach to group recommendation by minimizing user diversity in group formation and incorporating user contributions to the group preferences. We proposed two novel methods for generating user latent feature vectors to predict user similarity, a crucial step in user clustering. The user vector in the first method is obtained from a trained Matrix Factorization (MF) model, while in the second method, it is derived from the weights of a trained Graph Convolutional Network (GCN). Both methods, which operate on pairwise preference data, enhance the accuracy of user clustering compared to conventional single rating data. Building on these similarity predictions, we introduce diversity-based clustering techniques using a greedy algorithm that minimizes diversity scores, resulting in groups of users with similar preferences. This significantly improved the accuracy and fairness of recommendations by suggesting items likely to appeal to the majority within each group. Additionally, we propose a Group Recommendation System (GRS) that quantifies each user's contribution to the group's preferences using a game theory concept named Shapley value. Our extensive testing, including various clustering approaches demonstrated the superiority of our method over existing state-of-the-art techniques. 
The following figure illustrates the process of using Shapley value for making group recommendations:
<p align="center">
<img style="width: 80%;" src="https://github.com/RozaAbolghasemi/Shaply_GRS/blob/main/Shapley_GRS3.png">
</p>

## Execution Dependencies
We are using pandas, numpy, and sklearn. Install them by running:
```
pip install numpy
pip install pandas
pip install sklearn
```
The hyperparameters including group sizes, and number of top items can be adjusted to yeild the best results.


### Dataset
* Food dataset: The project uses a food dataset from an online experiment, Consens@OsloMet, at Oslo Metropolitan University (Norway), focusing on group food preferences. The experiment, approved by the Norwegian Centre for Research Data, used an online interface where experts provided pairwise scores for food pairs, displayed as probability scores.
* [Car dataset](https://users.cecs.anu.edu.au/~u4940058/CarPreferences.html): The dataset, provided by Abbasnejad et al. in 2013, includes car preferences from 60 U.S. users via Amazon's Mechanical Turk. It covers ten cars, with each user comparing all 45 possible pairs, yielding 90 observations per user. Besides pairwise preference scores, it contains files on user attributes (education, age, gender, region) and car attributes (body type, transmission, engine capacity, fuel).

* ----------------------------------------------------------------------

**License**

[MIT License](https://github.com/RozaAbolghasemi/Shaply_GRS/blob/main/LICENSE)


----------------------------------------------------------------------
