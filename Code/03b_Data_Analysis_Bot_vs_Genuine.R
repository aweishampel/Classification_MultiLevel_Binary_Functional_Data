##################
#Data Analysis for "Classification of Social Media Users Using a Generalized Functional Analysis" 
#
#Authors: Anthony Weishampel, Ana-Maria Staicu, Bill Rand
#Date: 7/16/2024
#
#Notes: Make sure to compile the 01_functions.R file first
#      These results are for the Bot Detection Analysis of Table 2
##################

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
numCores_to_run = 1

#Get Number for iterations
ITER = 2

#Format all data
# this is not the number of days wanted to analyze
J = J_train


set.seed(1234)


core_function = function(i){
  
  #i = 1
  #set seed for each iteration needed because code is run on multiple cores
  set.seed(i)
  
  #matrix to record accuracies
  acc_mat = matrix(NA, nrow = 5,  ncol = 1)
  f1_mat  = matrix(NA, nrow = 5,  ncol = 1)
  
  if(i%%10 == 0 || i==1){
    print(paste("On Iteration ", i , " out of ", ITER))
  }
  
  #only select the first J days
  J = J_train
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
  
  #start_time = 1
  
  timelines_genuine2 = timelines_genuine
  timelines_bots2 = timelines_bots
  timelines_fs1_2 = timelines_fs1
  
  #only include accounts which were active during those J days
  # can remove if you want all accounts but run into issues with GLM and GAM.
  min_num_tweets = 1
  
  timelines_genuine2 = timelines_genuine2[rowSums(timelines_genuine2)>min_num_tweets,]
  timelines_bots2 = timelines_bots2[rowSums(timelines_bots2)>min_num_tweets,]
  timelines_fs2_2 = timelines_fs1_2[rowSums(timelines_fs1_2)>min_num_tweets,]
  
  
  #select training and testing sets for genuine accounts
  N_genuine = dim(timelines_genuine2)[1]
  p_train = 0.8
  N_gtrain = round(N_genuine * p_train)
  train_genuine = sample(1:N_genuine, N_gtrain)
  Curves_gtrain = timelines_genuine2[train_genuine, ]
  Curves_gtest = timelines_genuine2[-train_genuine, ]

  #select training and testing sets for bots
  N_bots = dim(timelines_bots2)[1]
  N_btrain = round(N_bots * p_train)
  train_bots = sample(1:N_bots, N_btrain)
  Curves_btrain = timelines_bots2[train_bots, ]
  Curves_btest = timelines_bots2[-train_bots, ]

  #select training and testing sets for fs2
  N_r = dim(timelines_fs2_2)[1]
  N_rtrain = round(N_r * p_train)
  train_r = sample(1:N_r, N_rtrain)
  Curves_FStrain = timelines_fs2_2[train_r, ]
  Curves_FStest = timelines_fs2_2[-train_r, ]

  
  Curves_train = rbind(Curves_gtrain, Curves_btrain)
  Curves_test = rbind(Curves_gtest, Curves_btest)

  #D = dim(Curves_train)[2]
  
  #Two Levels Levels Can easily change to 4 levels by changing the levels
  Classes_train = c(rep(1, N_gtrain), rep(2, N_btrain))
  Classes_train = as.factor(Classes_train)
  Classes_test = c(rep(1, N_genuine-N_gtrain), rep(2, N_bots-N_btrain))
  Classes_test = as.factor(Classes_test)
  
  
  Curves_train = Curves_train[ , start_time:D ]
  Classes_train = Classes_train[rowSums(Curves_train)>min_num_tweets]
  Curves_train = Curves_train[rowSums(Curves_train)>min_num_tweets, ]
  
  #print(table(Classes_train))
  
  Curves_test = Curves_test[ , start_time_test:D_test ]
  Classes_test = Classes_test[rowSums(Curves_test)>min_num_tweets]
  Curves_test = Curves_test[rowSums(Curves_test)>min_num_tweets, ]
  
  #print(table(Classes_test))
  
  individuals = 1:length(Classes_train)
  N_train = length(Classes_train)
  N_test = length(Classes_test)
  
  tt=seq(0,1, len=D)
  
  Curves_train2 = Curves_train
  Curves_test2 = Curves_test
  
  #convert to multilevel_framework
  D = 48
  #dim(Curves_train)
  #dim(Curves_test)
  Curves_train = t(matrix(t(Curves_train), nrow = D))
  Curves_test = t(matrix(t(Curves_test), nrow = D))
  
  posting_days = 1-(rowSums(Curves_train)==0)
  s_mat_train = t(matrix(as.numeric(matrix(posting_days, nrow = J_train)), nrow = J_train))
  
  posting_days = 1-(rowSums(Curves_test)==0)
  s_mat_test = t(matrix(as.numeric(matrix(posting_days, nrow = J_test)), nrow = J_test))
  
  Js_s_train = rowSums(s_mat_train)
  Js_s_test = rowSums(s_mat_test)
  
  #Get the parsimonious distribution
  
  cur.train.linear = multilevel_linear_fpca(Curves_train, J,
                                            pve1 = 0.95, pve2 = 0.75)
  
  mu_t_hat = cur.train.linear$mu_hat
  eigen_vals1 = cur.train.linear$eigen_vals1
  eigen_funcs1 = cur.train.linear$eigen_funcs1
  eigen_vals2 = cur.train.linear$eigen_vals2
  eigen_funcs2 = cur.train.linear$eigen_funcs2
  
  
  posting_days = (rowSums(Curves_train)>1)
  s_mat_hat_train = t(matrix(as.numeric(matrix(posting_days, nrow = J)), nrow = J))
  
  users_to_keep = which(rowSums(s_mat_hat_train>0)>1)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Curves_train2 = Curves_train2[users_to_keep, ]
  Classes_train = Classes_train[users_to_keep]
  s_mat_train = s_mat_hat_train[users_to_keep, ]

  Curves_train = Curves_train[rows_to_keep,]
  N_train = length(Classes_train)
  users_to_keep_train = users_to_keep
  
  scores_train = estimate_scores(Curves_train, s_mat = s_mat_train, I=N_train,  J=J,
                                 eigen_vals1, eigen_vals2,
                                 eigen_funcs1, eigen_funcs2, mu_t_hat)
  
  
  posting_days = (rowSums(Curves_test)>1)
  s_mat_hat_test = t(matrix(as.numeric(matrix(posting_days, nrow = J_test)), nrow = J_test))
  
  #set to 0 we dont want testing set size to change
  min_num_test = 0
  J_test_max = J_test
  users_to_keep = which(rowSums(s_mat_hat_test>0)>min_num_test)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J_test+1):((x)*J_test)))
  Curves_test2 = Curves_test2[users_to_keep, ]
  Classes_test = Classes_test[users_to_keep]
  s_mat_test = s_mat_hat_test[users_to_keep, ]
  Curves_test = Curves_test[rows_to_keep,]
  Ys_train_reduced = Classes_train
  Ys_test_reduced = Classes_test
  users_to_keep_test = users_to_keep
  
  N_test = length(Classes_test)
  
  #estimate scores testing set
  scores_test=estimate_scores(Curves_test, s_mat = s_mat_test, I=N_test, J=J_test,
                              eigen_vals1, eigen_vals2,
                              eigen_funcs1, eigen_funcs2, mu_t_hat)
  
  #print(table(Ys_train_reduced))
  
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  s_mat_hat_test[users_to_keep_test, 1:J_test_max],
                          s_mat_hat_train =  s_mat_hat_train[users_to_keep_train,])
  
  #clock the time
  #Determine how well the classifier performed
  t1 = table(factor(guess, levels = unique(Ys_test_reduced)), Ys_test_reduced)
  #record results for proposed method
  acc_mat[1, 1] = sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  # sens_mat[1,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  # spec_mat[1,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  # precision_mat[1,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  f1_mat[1, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1)
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  
  #####
  #Comparision Method Scores only
  #####
  
  guess = nb_updated_grid_scores_only(scores = scores_train, classes = Ys_train_reduced,
                                      prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                                      scores_test =  scores_test)
  
  t2 = table(factor(guess, levels = unique(Ys_test_reduced)), Ys_test_reduced)
  acc_mat[3, 1] = sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  f1_mat[3, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1)
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  
  #Run random forest classifier
  
  D2 = J_test*48
  tt=seq(0,1, len=D2)
  X_dat_s = as.matrix(Curves_train2[,1:D2])
  X_dat_s_test = as.matrix(Curves_test2[,1:D2])
  
  rf1 = randomForest(y = Ys_train_reduced, x = X_dat_s, ntree = 100)
  guess = apply(X_dat_s_test, 1, function(x) predict(rf1, x))
  t4 = table(factor(guess, levels = unique(Classes_test)), Classes_test)
  acc_mat[4,1] = sum(diag(t4))/sum(t4)
  Ys_test_reduced = Classes_test
  f1_mat[4, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1)
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  if(J_train == J_test){
    ####
    #Single Level
    ####
    
    #get start time
    st = Sys.time()
    D2 = J*D
    tt=seq(0,1, len=D2)
    
    ##
    #Step 1 of the proposed method
    ##
    N = dim(X_dat_s)[1]
    vec = matrix(1:(N), ncol = 1)
    smoothed_x = logit(t(apply(vec, 1, function(x) regression_g(x, X_dat_s, tt, k=J))))
    
    ##
    #Step 2 of the proposed method
    ##
    fpca.cur2 = fpca.face(smoothed_x, pve = 0.98, p=3, m=2, knots = J) #lambda selected via grid search optim, #p=degree of splines
    #correct fpca.cur2 eigenval values b/c too large
    #get multiplier is function of D in fact its 1/D
    #get_multiplier = sum((fpca.cur2$efunctions[,1])^2/length(fpca.cur2$efunctions[,1]))
    get_multiplier = 1/D2
    fpca.cur = fpca.cur2
    #correct eigenfunctions
    fpca.cur$efunctions = fpca.cur2$efunctions/sqrt(get_multiplier)
    #correct eigenvalues
    fpca.cur$evalues = fpca.cur2$evalues*get_multiplier
    #correct scores
    fpca.cur$scores = fpca.cur2$scores*sqrt(get_multiplier)
    
    ##
    #STEP 3:
    # Set up and apply Bayesglm framework
    ##
    fit = list(mu = fpca.cur$mu,
               evalues = fpca.cur$evalues,
               efunctions = fpca.cur$efunctions)
    
    mu_t_hat = fit$mu
    eigen_vals1 = fit$evalues
    eigen_funcs1 = fit$efunctions
    
    #data frame used in bayesglm
    dta = data.frame(index = rep(tt, N),
                     value = c(t(X_dat_s)),
                     id = rep(1:N, each = D2))
    
    npc = length(eigen_vals1)
    if(npc>1){
      for (z in 1:npc) {
        dta <- cbind(dta, rep(eigen_funcs1[,z], N))
      }
    }else{
      dta = cbind(dta, matrix(eigen_funcs1, ncol =1))
    }
    
    #assign names to data frame
    names(dta)[4:(4 + npc - 1)] <- c(paste0("psi", 1:npc))
    #repeat mean function in data frame once per user
    dta$mu = rep(mu_t_hat , N)
    
    #get formula for glm
    glm_structure = paste(paste0("psi", 1:npc), collapse = "+")
    glm_structure = paste("value ~ -1 + offset(mu) +" , glm_structure , sep="")
    #set scale for the glm
    prior_scales_test = eigen_vals1
    
    #Estimate the Scores for the training set
    vec = matrix(1:N, ncol = 1)
    #vec = matrix(vec[users_to_keep_train,], ncol = 1)
    scores_train = t(apply(vec, 1, function(x) regression_bf2(x, dta, glm_structure, prior_scales_test)))
    
    #Step 3 for Testing Data
    #Get the socres for the testing set
    
    #just like before define data frame
    N_test = dim(X_dat_s_test)[1]
    dta = data.frame(index = rep(tt, N_test),
                     value = c(t(X_dat_s_test)),
                     id = rep(1:N_test, each = D2))
    
    npc = length(eigen_vals1)
    
    if(npc>1){
      for (z in 1:npc) {
        dta <- cbind(dta, rep(eigen_funcs1[,z], N_test))
      }
    }else{
      dta = cbind(dta, matrix(eigen_funcs1, ncol =1))
    }
    names(dta)[4:(4 + npc - 1)] <- c(paste0("psi", 1:npc))
    dta$mu = rep(mu_t_hat , N_test)
    
    glm_structure = paste(paste0("psi", 1:npc), collapse = "+")
    glm_structure = paste("value ~ -1 + offset(mu) +" , glm_structure , sep="")
    
    vec = matrix(1:N_test, ncol = 1)
    scores_test = t(apply(vec, 1, function(x) regression_bf2(x, dta, glm_structure, prior_scales_test)))
    
    #step 4
    #get propability of being in each group
    Ys_train = Classes_train
    prior_g = c(table(Ys_train)/length(Ys_train))
    #run non parametric bayes classifier
    guess = nb_updated_grid_scores_only(scores_train,
                                        Ys_train,
                                        prior_g, scores_test,
                                        min.h = 0.3, max.h = 1.5)
    
    #clock the time
    et = Sys.time()
    time_diff = et - st
    
    #Determine how well the classifier performed
    Ys_test_reduced = Classes_test
    t3 = table(guess, Ys_test_reduced)
    acc_mat[5,1] = sum(diag(t3))/sum(t3)
    f1_mat[5, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                                sum(guess==1 & Ys_test_reduced == 2)))^(-1)
                       +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                                 sum(guess==1 & Ys_test_reduced == 1)))^(-1))
    
    
  }
  
  # return list
  return_list = list()
  return_list[[1]] = c(acc_mat) #accuracy
  return_list[[2]] = t1 #confusion matrix for proposed method
  return_list[[3]] = t4 #confusion matrix for Scores Only
  #return_list[[4]] = t3 #confusion matrix for single level
  return_list[[5]] = c(f1_mat) #confusion matrix for random forest
  return(return_list)
  
}



#J = J_for_analysis
#J times number of observations in a day
#D = J*(60*24)/num_mins
#apply core function to each core to parallelize the analysis
results = mclapply(1:ITER, function(x) core_function(x), mc.cores = numCores_to_run)

print(results)

#get results from the previous apply function
results1 = list()
results2 = list()
results3 = list()
results4 = list()
results5 = list()

for(i in 1:ITER){
  results1[[i]] = results[[i]][[1]]
  results2[[i]] = results[[i]][[2]]
  results3[[i]] = results[[i]][[3]]
  #results4[[i]] = results[[i]][[4]]
  results5[[i]] = results[[i]][[5]]
  
}

#format results & print them
results = matrix(unlist(results1), ncol = ITER)
print(results)

results2 = matrix(unlist(results2), ncol = ITER)
print(results2)

results3 = matrix(unlist(results3), ncol = ITER)
print(results3)

#results4 = matrix(unlist(results4), ncol = ITER)
#print(results4)

results5 = matrix(unlist(results5), ncol = ITER)
print(results5)

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
#print(apply(results4, 1, function(x) mean(x, na.rm = T)))
#print(apply(results4, 1, function(x) sd(x, na.rm = T)/sqrt(ITER)))

#Print mean and sd
print(apply(results5, 1, function(x) mean(x, na.rm = T)))
print(apply(results5, 1, function(x) sd(x, na.rm = T)/sqrt(ITER)))




