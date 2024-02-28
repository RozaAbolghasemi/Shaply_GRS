
import pandas as pd
import numpy as np
import csv
import os
from collections import defaultdict
import random
from sklearn.metrics import roc_auc_score
#import scores
import matplotlib.pyplot as plt
#
##%%         ****************** DATA3 *******************
#
Movie_num = 10
num_OriginalUsers = 60
num_generatedSamples = 0
User_num = num_OriginalUsers + num_generatedSamples
Data = np.zeros((User_num, Movie_num,Movie_num))

#We just considered the two forst datas. The two last are removed due to repeatition

CarData = pd.read_csv("/Users/rozaabol/Desktop/Paper2_GDM/Codes/CarDataset/prefs1.csv")

TopMovies = range(1, Movie_num+1)

Data = np.empty((User_num,Movie_num,Movie_num))
for index, row in CarData.iterrows():
    if row.User_ID <=num_OriginalUsers:
        user = row.User_ID    #CarData.User_ID[index]
        item1 = row.Item1_ID    #CarData.Item1_ID[index]
        item2 = row.Item2_ID    #CarData.Item2_ID[index]
        Data [user-1][item1-1][item2-1]= 1
        Data [user-1][item2-1][item1-1]= 0
        Data [user-1][item1-1][item1-1]= 0.5
        Data [user-1][item2-1][item2-1]= 0.5

TopUsers= range(User_num)
TopItems= range(Movie_num)


#%%

Num_MV = 45 * User_num - 1 # Number of Missing Values
Num_Repeat = 10 # Number of Repeats

#if not os.path.isfile('Ranking Error Viedma.csv'):
MissingValues = list(range(1, Num_MV+1))
Repeats = list(range(1, Num_Repeat+1))
DF_Error_pairscore_Entropy = pd.DataFrame(index=Repeats, columns=MissingValues) #df containing Number of Missing Values and Number of Repeats
DF_Error_Ranking_Entropy = pd.DataFrame(index=Repeats, columns=MissingValues) #df containing Number of Missing Values and Number of Repeats

epoch =1
user_count =  User_num
item_count =  Movie_num
num_Run = Num_MV 
#predicted_pairscores_run = np.empty((num_Run, user_count, item_count, item_count))
predicted_pairscores_run = np.empty((Num_Repeat, user_count, item_count, item_count))




for rr in range(Num_Repeat):
    
    # RemoveItemsList is list of data that are going to be removed from the training set.  
    # Num_RemivedItems=num_Run    In the forst run we have one removed item and it addes one for every iteraton.

    RemoveItems = pd.DataFrame(columns=['User', 'Movie1','Movie2'])
    
    # To make unique random numbers:
    n=0
    item1 = np.random.randint(item_count-1)
    item2 = np.random.randint(item1+1, item_count)
    user = np.random.randint(user_count)
    RemoveItems.loc[len(RemoveItems)] = [user, item1 , item2]
    
    while n < num_Run: 
        item1 = np.random.randint(item_count-1)
        item2 = np.random.randint(item1+1, item_count)
        user = np.random.randint(user_count)
        if (((RemoveItems['User'] == user) & (RemoveItems['Movie1'] == item1)& (RemoveItems['Movie2'] == item2)).any()) == False:
            RemoveItems.loc[len(RemoveItems)] = [user, item1 , item2]
            n += 1 


#%%
#Matrix Factorization


    Error_pairscore_Entropy  = np.zeros((num_Run)) 
    Error_Ranking_Entropy  = np.zeros((num_Run)) 
    MPlist = pd.DataFrame(index=range(num_Run), columns=TopMovies) #np.zeros((item_count, num_Run))
    MRlist = pd.DataFrame(index=range(num_Run), columns=TopMovies) #np.zeros((item_count, num_Run))  
    
    k=2  #len of embeddings
    lr=0.01  
    reg = 0.01
    n_epoch = 100
    
    
    for run in range(1,num_Run+1):
        if run % 50 ==0:
            print("repeat:", rr, "   Run:", run, "   Epoch:", epoch) # 10, 1600, 100
                          
        biasV = np.random.rand(item_count) * 0.01  
        # Initialize the embedding weights.
        U = np.random.rand(user_count, k) * 0.01  
        V = np.random.rand(item_count, k) * 0.01  

        for epoch in range(n_epoch): 
#            if epoch%1000 == 0:
#                print("repeat:", rr, "   Run:", run, "   Epoch:",epoch)
            for u in range(user_count):
                for i in range(item_count):
                    for j in range(item_count):
                        if [u, i, j] not in RemoveItems.values[:run].tolist():
                            r_uij = Data[u][i][j]
#                            if Data[u][i][j] > 0:
#                                r_uij = 1
#                            elif Data[u][i][j] < 0:
#                                r_uij = 0
#                            else:
#                                r_uij = 0.5 
                                                    
                            # Update weights by gradients.  
                            rp_ui = np.dot(U[u], V[i].T) + biasV[i] #rp is predicted rating
                            rp_uj = np.dot(U[u], V[j].T) + biasV[j]
                            rp_uij = rp_ui - rp_uj # rp is predicted pairwise rating
                
                            loss_func =  1.0 / (1 + np.exp(-rp_uij)) - r_uij  #-1.0 / (1 + np.exp(rp_uij))
                            
                              # update U and V
                            U[u] += -lr * (loss_func * (V[i] - V[j]) + reg * U[u]) #I write it according to BPR. Is it correct?????
                            
                            if r_uij >0: #prefered item must increse and less prefer one, decrease.
                                V[i] += -lr * (loss_func * U[u] + reg * V[i])
                                V[j] += -lr * (loss_func * (-U[u]) + reg * V[j])
                                # update biasV
                                biasV[i] += -lr * (loss_func + reg * biasV[i])
                                biasV[j] += -lr * (-loss_func + reg * biasV[j]) 
                            else:
                                V[j] += -lr * (loss_func * U[u] + reg * V[j])
                                V[i] += -lr * (loss_func * (-U[u]) + reg * V[i])       
                                # update biasV
                                biasV[j] += -lr * (loss_func + reg * biasV[j])
                                biasV[i] += -lr * (-loss_func + reg * biasV[i]) 
               
            
        UserEmbedding = U
        MovieEmbedding = V
        
        
        #*********** Evaluation: ************
#        for u in range(user_count):
#            for i in range(item_count):
#                for j in range(item_count):
#                    predict_scores[u][i] += r[u][i][j]
#                predict_scores[u][i] /= item_count
        
        predict_scores = np.mat(UserEmbedding) * np.mat(MovieEmbedding.T)+ biasV
        PredictMatrix = pd.DataFrame(predict_scores, index=TopUsers, columns=TopMovies)   
        # I normalize the predicted output to [0,1] range, so it will be closer to the missing values range
        max_predicted = PredictMatrix.max().max() #### New update :)
        min_predicted = PredictMatrix.min().min() #### New update :)
        PredictMatrix_normalized = (PredictMatrix - min_predicted)/ (max_predicted - min_predicted) #### New update :)
        movies_predicted_order = PredictMatrix_normalized.mean(axis=0).sort_values(ascending=False).index 
        
        Original_Data = np.zeros((user_count, item_count))
        for u in range (user_count):
            Original_Data[u][:]= np.mean(Data[u], axis=1)
        RealMatrix = pd.DataFrame(Original_Data, index=TopUsers, columns=TopMovies)  #1: Real order of the full matrix    
        max_Real = RealMatrix.max().max() #### New update :)
        min_Real = RealMatrix.min().min() #### New update :)
        RealMatrix_normalized = (RealMatrix - min_Real)/ (max_Real - min_Real) #### New update :)
        movies_real_order =  RealMatrix_normalized.mean(axis=0).sort_values(ascending=False).index           
#        if run == 1: #3: order of the matrix after removing the first item(first fully matrix after 1 experiment with the trained U and V)  
#             movies_real_order = movies_predicted_order        
        
        
        ##****************  Errooooor:
        predicted_pairscores = np.zeros((user_count,item_count,item_count))
        for u in range(user_count):
            for i in range(item_count):
                for j in range(item_count):
                        rp_ui = np.dot(U[u], V[i].T) + biasV[i] 
                        rp_uj = np.dot(U[u], V[j].T) + biasV[j]
                        rp_uij = rp_ui - rp_uj
                        predicted_pairscores[u][i][j] = 1.0 / (1 + np.exp(-rp_uij))
                        
        SumDif = 0                
        for [u, i, j] in RemoveItems.values[:run].tolist():
            SumDif += abs(predicted_pairscores[u][i][j] - Data[u][i][j])
        
        Error_pairscore_Entropy[run-1] = SumDif/num_Run  
        #predicted_pairscores_run[run-1] = predicted_pairscores
        
  
            
        #%%
        
        # Recover the group rank when we have missing values : true ranking vs ranking of predicted matrix:
        predict_pairscores_Entropy = np.zeros((user_count, item_count, item_count))   
        for u in range(user_count):
            for i in range(item_count):
                for j in range(item_count):    
                    if [u, i, j] in RemoveItems.values[:run].tolist():
                        predict_pairscores_Entropy[u][i][j] = predicted_pairscores[u][i][j]
                    else:
                        predict_pairscores_Entropy[u][i][j] = Data[u][i][j]
        
        predict_scores_Entropy = np.zeros((user_count, item_count)) 
        for u in range (user_count):
            predict_scores_Entropy[u][:]= np.mean(predict_pairscores_Entropy[u], axis=1)  
            
        PredictMatrix_Entropy = pd.DataFrame(predict_scores_Entropy, index=TopUsers, columns=TopMovies)        
        predicted_order_Entropy = PredictMatrix_Entropy.mean(axis=0).sort_values(ascending=False).index
        
        Original_Data = np.zeros((user_count, item_count))
        for u in range (user_count):
            Original_Data[u][:]= np.mean(Data[u], axis=1)
        RealMatrix = pd.DataFrame(Original_Data, index=TopUsers, columns=TopMovies)  #1: Real order of the full matrix    
        real_order =  RealMatrix.mean(axis=0).sort_values(ascending=False).index 
        
        gScore_Entropy = PredictMatrix_Entropy.mean(axis=0)
        gScore_InputData =  RealMatrix.mean(axis=0)
        #%%
        
        Error=0
        k=0
        mpList = predicted_order_Entropy.tolist()
        mrList = real_order.tolist()
        for mm in mpList:
           Error += abs(mrList.index(mm) - k)
           k+= 1
        Error_Entropy = float(Error/item_count)
        
        Error_Ranking_Entropy[run-1] = Error_Entropy
        
    predicted_pairscores_run[rr] = predicted_pairscores     
        
    DF_Error_pairscore_Entropy.at[rr+1,:] = Error_pairscore_Entropy
    DF_Error_Ranking_Entropy.at[rr+1,:] = Error_Ranking_Entropy

DF_Error_pairscore_Entropy.to_csv(path_or_buf="/Users/rozaabol/Desktop/Paper2_GDM/Codes/Error_Prediction_Entropy.csv")
DF_Error_Ranking_Entropy.to_csv(path_or_buf="/Users/rozaabol/Desktop/Paper2_GDM/Codes/Error_Ranking_Entropy.csv")


