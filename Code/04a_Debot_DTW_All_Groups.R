##################
#Debot Analysis for "Classification of Social Media Users Using a Generalized Functional Analysis"
#
#Authors: Anthony Weishampel, Ana-Maria Staicu, Bill Rand
#Date: 7/16/2024
#
#Notes: Make sure to compile the 01_functions.R file first
##################


library(dtw)
library(parallel)
library(readr)
library(randomForest)

#load each dataset...make sure to change the directory to the location of the data
timelines_genuine <- read_csv("/dir/timelines_genuine.csv")
timelines_bots <- read_csv("/dir/timelines_bots.csv")
timelines_fs1<- read_csv("/dir/timelines_fs1.csv")

# number of days * number of 5 minute long intervals in a day
D = 50*24*2+1

#Format data for each group
timelines_genuine2 = timelines_genuine[,2:D]
timelines_bots2 = timelines_bots[,2:D]
timelines_fs12 = timelines_fs1[,2:D]

#format to matrix
timelines_genuine = data.matrix(timelines_genuine2)
timelines_bots = data.matrix(timelines_bots2)
timelines_fs1 = data.matrix(timelines_fs12)

#make sparse for easier storage
timelines_genuine = Matrix(timelines_genuine, sparse=T)
timelines_bots = Matrix(timelines_bots, sparse=T)
timelines_fs1 = Matrix(timelines_fs1, sparse=T)


#Number of days to analyze
#Need to change for all of the different combinations 
J_train = 14
J_test = 14

#number of minutes per break
num_mins = 30

#random_days
#variable to select random starting day (used in days sensitivity analysis)
Random_start_day = T
Random_end_day  = T

#Number of cores able to run code on
numCores_to_run = 16

#Get Number for iterations
ITER = 50

#Format all data
# this is not the number of days wanted to analyze
J = J_train


set.seed(1234)


dtw_KNN = function(k=5, x, binary_train, classes_train){
  
  dtws = dtwDist(binary_train, matrix(x, nrow= 1))
  k_val = sort(dtws)[k]
  dtws_under_k = which(dtws <= k_val)
  k_classes = classes_train[dtws_under_k]
  guess = names(which.max(table(k_classes)))
  return(guess)
  
}

dtw_KNN_total = function(k=5, x, binary_train, classes_train){
  dtws = dtwDist(binary_train, matrix(x, nrow= 1))
  k_val = order(dtws)[1:k]
  k_classes = classes_train[k_val]
  return(k_classes)
}



core_function = function(i, k_val=5){
  
  set.seed(i)
  
  acc_mat = matrix(NA, nrow = 1,  ncol = 1)
  precision_mat = matrix(NA, nrow = 1,  ncol = 1)
  sens_mat = matrix(NA, nrow = 1,  ncol = 1)
  spec_mat = matrix(NA, nrow = 1,  ncol = 1)
  f1_mat = matrix(NA, nrow = 1,  ncol = 1)
  
  if(i%%25 == 0 || i==1){
    print(paste("On Iteration ", i , " out of ", ITER))
  }
  
  ###
  #DTW Method
  ###
  
  
  
  #only select the first 14 days
  J = J_train
  num_mins = 30
  #J times number of observations in a day
  D = J*(60*24)/num_mins
  D_test = J_test*(60*24)/num_mins
  nd_train = 50 - J - 1
  nd_test = 50 - J_test - 1
  
  if(Random_start_day){
    start_time = (sample((1:nd_train),1)-1)*48+1
    start_time_test = start_time
  }else{
    start_time = 1
    start_time_test = 1
  }
  D  = start_time + D - 1
  
  if(Random_end_day){
    start_time_test = (sample((1:nd_test),1)-1)*48+1
  }
  
  D_test  = start_time_test + D_test - 1
  
  #print("test1")
  
  
  timelines_genuine = timelines_genuine2
  timelines_bots = timelines_bots2
  timelines_fs1 = timelines_fs1
  
  N_genuine = dim(timelines_genuine)[1]
  p_train = 0.8
  N_gtrain = round(N_genuine * p_train)
  train_genuine = sample(1:N_genuine, N_gtrain)
  Curves_gtrain = timelines_genuine[train_genuine, ]
  Curves_gtest = timelines_genuine[-train_genuine, ]
  
  N_bots = dim(timelines_bots)[1]
  N_btrain = round(N_bots * p_train)
  train_bots = sample(1:N_bots, N_btrain)
  Curves_btrain = timelines_bots[train_bots, ]
  Curves_btest = timelines_bots[-train_bots, ]
  
  N_r = dim(timelines_fs1)[1]
  N_rtrain = round(N_r * p_train)
  train_r = sample(1:N_r, N_rtrain)
  Curves_rtrain = timelines_fs1[train_r, ]
  Curves_rtest = timelines_fs1[-train_r, ]
  
  
  Curves_train = rbind(Curves_gtrain, Curves_btrain, Curves_FStrain)
  Curves_test = rbind(Curves_gtest, Curves_btest, Curves_FStest)
  
  #D = dim(Curves_train)[2]
  
  #Two Levels Levels Can easily change to 4 levels by changing the levels
  Classes_train = c(rep(1, N_gtrain), rep(2, N_btrain), rep(3, N_rtrain))
  Classes_train = as.factor(Classes_train)
  Classes_test = c(rep(1, N_genuine-N_gtrain), rep(2, N_bots-N_btrain), rep(3, N_r-N_rtrain))
  Classes_test = as.factor(Classes_test)
  
  
  individuals = 1:length(Classes_train)
  N_train = length(Classes_train)
  N_test = length(Classes_test)
  
  min_num_tweets = 1
  
  Curves_train = Curves_train[ , start_time:D ]
  #static_train = static_train[rowSums(Curves_train)>min_num_tweets, ]
  Classes_train = Classes_train[rowSums(Curves_train)>min_num_tweets]
  Curves_train = Curves_train[rowSums(Curves_train)>min_num_tweets, ]
  
  #print(table(Classes_train))
  
  Curves_test = Curves_test[ , start_time_test:D_test ]
  #static_test = static_test[rowSums(Curves_test)>min_num_tweets, ]
  Classes_test = Classes_test[rowSums(Curves_test)>min_num_tweets]
  Curves_test = Curves_test[rowSums(Curves_test)>min_num_tweets, ]
  
  print(dim(Curves_train))
  #length(Classes_train)
  
  individuals = 1:length(Classes_train)
  N_train = length(Classes_train)
  N_test = length(Classes_test)
  
  #set number of accounts in training set 
  #Needs to be smaller because of large computation time. 
  N_train = 100
  training_accounts = sample(1:length(Classes_train), N_train)
  Classes_train = Classes_train[training_accounts]
  Curves_train = Curves_train[training_accounts,]
  
  #smaller test size to find the kl
  #N_test = 100
  training_accounts = sample(1:length(Classes_test), N_test)
  Classes_test = Classes_test[training_accounts]
  Curves_test = Curves_test[training_accounts,]
  

  pred_classes = apply(Curves_test, 1, function(x) dtw_KNN(k=k_val, x, Curves_train, Classes_train))
  t1 = table(pred_classes, Classes_test)
  #acc_mat[1,z_counter] = sum(diag(t1))/sum(t1)
  print(t1)
  
  acc_mat[1, 1] = sum(pred_classes==Classes_test)/length(Classes_test)
  sens_mat[1,1] = sum(pred_classes==1 & Classes_test == 1)/(sum(pred_classes==2 & Classes_test == 1)+sum(pred_classes==1 & Classes_test == 1))
  spec_mat[1,1] = sum(pred_classes==2 & Classes_test == 2)/(sum(pred_classes==2 & Classes_test == 2)+sum(pred_classes==1 & Classes_test == 2))
  precision_mat[1,1] = sum(pred_classes==1 & Classes_test == 1)/(sum(pred_classes= 1 & Classes_test == 1)+sum(pred_classes==1 & Classes_test == 2))
  f1_mat[1, 1]  = 2/((sum(pred_classes==1 & Classes_test == 1)/(sum(pred_classes== 1 & Classes_test == 1)+
                                                                  sum(pred_classes==1 & Classes_test == 2)))^(-1)
                     +(sum(pred_classes==1 & Classes_test == 1)/(sum(pred_classes==2 & Classes_test == 1)+
                                                                   sum(pred_classes==1 & Classes_test == 1)))^(-1))
  
  
  
  print(c(acc_mat, sens_mat, spec_mat, precision_mat))
  
  return_list = list()
  return_list[[1]] = c(acc_mat) #accuracy
  return_list[[2]] = t1 #confusion matrix for method
  return_list[[3]] = c(sens_mat) #confusion matrix for Scores Only
  #return_list[[4]] = c(spec_mat) #confusion matrix for single level
  return_list[[4]] = c(f1_mat) #confusion matrix for random forest
  return(return_list)
  
  #return(c(acc_mat, sens_mat, spec_mat, precision_mat))
  
}



get_k = function(k_min = 5, k_max = 20){
  
  set.seed(12)
  
  ###
  #DTW Method
  ###

  
  
  #only select the first 14 days
  J = J_train
  num_mins = 30
  #J times number of observations in a day
  D = J*(60*24)/num_mins
  D_test = J_test*(60*24)/num_mins
  nd_train = 50 - J - 1
  nd_test = 50 - J_test - 1
  
  if(Random_start_day){
    start_time = (sample((1:nd_train),1)-1)*48+1
    start_time_test = start_time
  }else{
    start_time = 1
    start_time_test = 1
  }
  D  = start_time + D - 1
  
  if(Random_end_day){
    start_time_test = (sample((1:nd_test),1)-1)*48+1
  }
  
  D_test  = start_time_test + D_test - 1
  
  #print("test1")

  
  timelines_genuine = timelines_genuine2
  timelines_bots = timelines_bots2
  timelines_fs1 = timelines_fs1
  
  N_genuine = dim(timelines_genuine)[1]
  p_train = 0.8
  N_gtrain = round(N_genuine * p_train)
  train_genuine = sample(1:N_genuine, N_gtrain)
  Curves_gtrain = timelines_genuine[train_genuine, ]
  Curves_gtest = timelines_genuine[-train_genuine, ]
  
  N_bots = dim(timelines_bots)[1]
  N_btrain = round(N_bots * p_train)
  train_bots = sample(1:N_bots, N_btrain)
  Curves_btrain = timelines_bots[train_bots, ]
  Curves_btest = timelines_bots[-train_bots, ]
  
  N_r = dim(timelines_fs1)[1]
  N_rtrain = round(N_r * p_train)
  train_r = sample(1:N_r, N_rtrain)
  Curves_rtrain = timelines_fs1[train_r, ]
  Curves_rtest = timelines_fs1[-train_r, ]
  
  
  Curves_train = rbind(Curves_gtrain, Curves_btrain, Curves_rtrain)
  Curves_test = rbind(Curves_gtest, Curves_btest, Curves_rtest)
  #D = dim(Curves_train)[2]
  
  #Two Levels Levels Can easily change to 4 levels by changing the levels
  Classes_train = c(rep(1, N_gtrain), rep(2, N_btrain), rep(3, N_rtrain))
  Classes_train = as.factor(Classes_train)
  
  Classes_test = c(rep(1, N_genuine-N_gtrain), rep(2, N_bots-N_btrain), rep(3, N_r-N_rtrain))
  Classes_test = as.factor(Classes_test)
  #D = dim(Curves_train)[2]
  
  individuals = 1:length(Classes_train)
  N_train = length(Classes_train)
  N_test = length(Classes_test)
  
  min_num_tweets = 1
  
  Curves_train = Curves_train[ , start_time:D ]
  #static_train = static_train[rowSums(Curves_train)>min_num_tweets, ]
  Classes_train = Classes_train[rowSums(Curves_train)>min_num_tweets]
  Curves_train = Curves_train[rowSums(Curves_train)>min_num_tweets, ]
  
  #print(table(Classes_train))
  
  Curves_test = Curves_test[ , start_time_test:D_test ]
  #static_test = static_test[rowSums(Curves_test)>min_num_tweets, ]
  Classes_test = Classes_test[rowSums(Curves_test)>min_num_tweets]
  Curves_test = Curves_test[rowSums(Curves_test)>min_num_tweets, ]
  
  #print(dim(Curves_train))
  #length(Classes_train)
  
  individuals = 1:length(Classes_train)
  N_train = length(Classes_train)
  N_test = length(Classes_test)
  
  #set number of accounts in training set 
  #Needs to be smaller because of large computation time. 
  N_train = 100
  training_accounts = sample(1:length(Classes_train), N_train)
  Classes_train = Classes_train[training_accounts]
  Curves_train = Curves_train[training_accounts,]

  #smaller test size to find the kl
  N_test = 100
  training_accounts = sample(1:length(Classes_test), N_test)
  Classes_test = Classes_test[training_accounts]
  Curves_test = Curves_test[training_accounts,]
  
  #dtw_KNN_first = function(Curves_test, Curves_train, Classes_train, Classes_test, k_min=3, k_max=20)
  ks = k_min:k_max
  accs = rep(NA, length(ks))
  
  pred_classes = apply(Curves_test, 1, function(x) dtw_KNN_total(k=k_max, x, Curves_train, Classes_train))
  pred_classes_mat = matrix(unlist(pred_classes), ncol = length(Classes_test))
  
  for(j in 1:length(ks)){
    
    k = ks[j]
    guesses = apply(pred_classes_mat[1:k,], 2, function(x) names(which.max(table(x))))
    accs[j] = sum(guesses==Classes_test)/length(Classes_test)
    
  }
  
  k=ks[which.max(accs)]
  
  print(k)
  return(k)

}



####
# Get Initial K value
####

k_val = get_k()
print(k_val)
#k_val = 8

results = mclapply(1:ITER, function(x) core_function(x, k_val), mc.cores = 16)
print(results)

#get results from the previous apply function
results1 = list()
results2 = list()
results3 = list()
results4 = list()

for(i in 1:ITER){
  results1[[i]] = results[[i]][[1]]
  results2[[i]] = results[[i]][[2]]
  results3[[i]] = results[[i]][[3]]
  results4[[i]] = results[[i]][[4]]

}

#format results & print them
results = matrix(unlist(results1), ncol = ITER)
print(results)

results2 = matrix(unlist(results2), ncol = ITER)
print(results2)

results3 = matrix(unlist(results3), ncol = ITER)
print(results3)

results4 = matrix(unlist(results4), ncol = ITER)
print(results4)

#Accuracy
#Print mean and sd
print(apply(results, 1, function(x) mean(x, na.rm = T)))
print(apply(results, 1, function(x) sd(x, na.rm = T)/sqrt(ITER)))

#Print mean and sd
print(apply(results2, 1, function(x) mean(x, na.rm = T)))
print(apply(results2, 1, function(x) sd(x, na.rm = T)/sqrt(ITER)))

#Print mean and sd
print(apply(results3, 1, function(x) mean(x, na.rm = T)))
print(apply(results3, 1, function(x) sd(x, na.rm = T)/sqrt(ITER)))

#Print mean and sd
print(apply(results4, 1, function(x) mean(x, na.rm = T)))
print(apply(results4, 1, function(x) sd(x, na.rm = T)/sqrt(ITER)))




