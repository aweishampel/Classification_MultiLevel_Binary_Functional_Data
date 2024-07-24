##################
#Simulation for Scenario A2 of 
#"Classification of Social Media Accounts Using a Generalized Multilevel Functional Analysis"
#
#Authors: Anthony Weishampel, Ana-Maria Staicu, Bill Rand
#Date: 7/16/2024
#
# Make sure to run 01_functions file first
##################


#Set scenario to generate the functional data
scenario_funcs = 3
#Set scenario to generate the S data
scenario_S     = 1 

#The number of points per series
D = 50

#define the grid
grid = seq(0, 1, length = D)

#Number of users in the testing sets
Ns = c(50, 200, 1000)

#Number of cores able to run code on
numCores_to_run = 16

#for parallel processing
numCores <- detectCores() # get the number of cores available

#number of iterations
ITER = 200

#number of curves in the testing set
N_test = 500

#set seed
set.seed(1234)

#number of curves 
Js = c(5, 25, 50)



####
#Function: Main Simulation function to pass through to different cores
#
#Input: 
# i: iteration to run simulation one
#
#Outputs: 
# misclassification, time, precision, sensitivity, and specificity for each tested classifier on iteration i
####
core_function = function(i2){
  
  #i2 = 15 
  #Need to seed for each different core to ensure same results 
  set.seed(i2)
  
  N = N2
  J = J
  D = D
  
  #matrices to return one row for each tested classifier
  misclass_mat = matrix(NA, nrow = 8,  ncol = 1)
  time_mat = matrix(NA, nrow = 8,  ncol = 1)
  precision_mat = matrix(NA, nrow = 8,  ncol = 1)
  sens_mat = matrix(NA, nrow = 8,  ncol = 1)
  spec_mat = matrix(NA, nrow = 8,  ncol = 1)
  f1_mat = matrix(NA, nrow = 8,  ncol = 1)
  
  i = i2
  #output to know how much is left
  if(i%%25 == 0 || i==1){
    print(paste("On Iteration ", i , " out of ", ITER))
  }
  
  grid = seq(0, 1, length = D)
  
  Ys_train = rbinom(N, 1, 0.5)
  #Ys_train = rbinom(N, 1, 1)
  
  X_tilde = generate_multilevel_data(scenario = scenario_funcs , grid = grid, 
                                     N=N, J=J, binary=T, 
                                     Ys = Ys_train, return_scores_too = T)
  V_scores = t(X_tilde$V_scores)
  X_tilde = X_tilde$Curves_binary
  #dim(X_tilde)
  
  #data_ls = generate_data_with_S(scenario = 1, X_tilde, N, J, Ys_train, 
  #                               alpha1 = 0.5, alpha2 = 0.5)
  data_ls = generate_data_with_S(scenario = scenario_S, X_tilde, N, J, Ys_train)
  
  #All Training & testing sets
  X_dat = data_ls$Curves_binary
  
  st = Sys.time()
  
  cur.train.linear = multilevel_linear_fpca(X_dat, J)
  
  mu_t_hat = cur.train.linear$mu_hat
  eigen_vals1 = cur.train.linear$eigen_vals1
  eigen_funcs1 = cur.train.linear$eigen_funcs1
  eigen_vals2 = cur.train.linear$eigen_vals2
  eigen_funcs2 = cur.train.linear$eigen_funcs2
  
  posting_days = (rowSums(X_dat)>1)
  s_mat_hat_train = t(matrix(as.numeric(matrix(posting_days, nrow = J)), nrow = J))
  
  users_to_keep = which(rowSums(s_mat_hat_train>0)!=0)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Ys_train_reduced = Ys_train[users_to_keep]
  scores_train = estimate_scores(X_dat[rows_to_keep,], 
                                 s_mat = s_mat_hat_train[users_to_keep,],
                                 I=length(users_to_keep),  J=J,
                                 eigen_vals1, eigen_vals2,
                                 eigen_funcs1, eigen_funcs2, mu_t_hat)
  users_to_keep_train = users_to_keep
  
  N_test = 200
  Ys_test = rbinom(N_test, 1, 0.5)
  X_tilde = generate_multilevel_data(scenario = scenario_funcs , grid = grid, 
                                     N=N_test, J=J, binary=T, 
                                     Ys = Ys_test, return_scores_too = T)
  V_scores = t(X_tilde$V_scores)
  X_tilde = X_tilde$Curves_binary
  
  #ways to generate S's
  #data_ls = generate_data_with_S(scenario = 1, X_tilde, N_test, J, Ys_test, 
  #                               alpha1 = 0.5, alpha2 = 0.5)
  data_ls = generate_data_with_S(scenario = scenario_S , X_tilde, N_test, J, Ys_test)
  
  #All Training & testing sets
  X_dat_test = data_ls$Curves_binary
  
  posting_days = (rowSums(X_dat_test)>1)
  s_mat_hat_test = t(matrix(as.numeric(matrix(posting_days, nrow = J)), nrow = J))
  
  
  users_to_keep = which(rowSums(s_mat_hat_test>0)!=0)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Ys_test_reduced = Ys_test[users_to_keep]
  scores_test = estimate_scores(X_dat_test[rows_to_keep,], 
                                s_mat = s_mat_hat_test[users_to_keep,],
                                I=length(users_to_keep),  J=J,
                                eigen_vals1, eigen_vals2,
                                eigen_funcs1, eigen_funcs2, mu_t_hat)
  users_to_keep_test = users_to_keep
  
  #estimate alpha_values too
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced+1,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  s_mat_hat_test[users_to_keep_test,],
                          s_mat_hat_train =  s_mat_hat_train[users_to_keep_train,])
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  Ys_test_reduced = Ys_test_reduced+1
  #Determine how well the classifier performed
  t1 = table(guess, Ys_test_reduced)
  #record results for proposed method
  misclass_mat[1, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[1,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[1,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[1,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[1, 1] = as.numeric(time_diff)
  f1_mat[1, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  ####
  #comparative method without S in classification 
  ####
  
  guess = nb_updated_grid_scores_only(scores = scores_train, classes = Ys_train_reduced+1,
                                      prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                                      scores_test =  scores_test)
  
  et = Sys.time()
  time_diff = et - st
  
  misclass_mat[2, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[2,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[2,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[2,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess= 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[2, 1] = as.numeric(time_diff)
  f1_mat[2, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  
  ###
  #Comparative method where we smooth alpha_hat over j
  ###
  
  
  alpha_hat_l = t(matrix(unlist(by(s_mat_hat_train[users_to_keep_train,], 
                                   Ys_train_reduced, colMeans)), nrow = J))
  
  alpha_hat_l = t(apply(alpha_hat_l, 1, function(x) gam(x~s(seq(0,1, length.out = J), bs = "cr", m=2, k=J),
                                                        family="gaussian", method = "REML")$fitted.values))
  
  #estimate alpha_values too
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced+1,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  s_mat_hat_test[users_to_keep_test,],
                          s_mat_hat_train =  s_mat_hat_train[users_to_keep_train,], 
                          alpha_js = alpha_hat_l)
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  #Ys_test_reduced = Ys_test_reduced+1
  #Determine how well the classifier performed
  t1 = table(guess, Ys_test_reduced)
  #record results for proposed method
  misclass_mat[3, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[3,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[3,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[3,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess= 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[3, 1] = as.numeric(time_diff)
  f1_mat[3, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  ####
  #comparative method with the single level approach
  ####
  
  #get start time
  st = Sys.time()
  D2 = J*D
  tt=seq(0,1, len=D2)
  
  X_dat_s = t(matrix(t(X_dat), nrow = D2))
  
  ##
  #Step 1 of the proposed method 
  ##
  vec = matrix(1:(N), ncol = 1)
  smoothed_x = logit(t(apply(vec, 1, function(x) regression_g(x, X_dat_s, tt, k=J))))
  
  ##
  #Step 2 of the proposed method
  ##
  fpca.cur2 = fpca.face(smoothed_x, pve = 0.95, p=3, m=2, knots = 2*J) #lambda selected via grid search optim, #p=degree of splines
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
  
  X_dat_s_test = t(matrix(t(X_dat_test), nrow = D2))
  
  #just like before define data frame 
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
  prior_g = c(table(Ys_train[users_to_keep_train])/length(Ys_train[users_to_keep_train]))
  #run non parametric bayes classifier
  pred_classes = nb_updated_grid_scores_only(scores_train[users_to_keep_train,], 
                                             Ys_train[users_to_keep_train]+1, 
                                             prior_g, scores_test[users_to_keep_test,], 
                                             min.h = 0.3, max.h = 1.5)
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  
  #Determine how well the classifier performed
  t1 = table(pred_classes, Ys_test_reduced)
  #record results for proposed method
  misclass_mat[4, 1] = 1- sum(pred_classes==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[4,1] = sum(pred_classes==1 & Ys_test_reduced == 1)/(sum(pred_classes==2 & Ys_test_reduced == 1)+sum(pred_classes==1 & Ys_test_reduced == 1))
  spec_mat[4,1] = sum(pred_classes==2 & Ys_test_reduced == 2)/(sum(pred_classes==2 & Ys_test_reduced == 2)+sum(pred_classes==1 & Ys_test_reduced == 2))
  precision_mat[4,1] = sum(pred_classes==1 & Ys_test_reduced == 1)/(sum(pred_classes= 1 & Ys_test_reduced == 1)+sum(pred_classes==1 & Ys_test_reduced == 2))
  time_mat[4, 1] = as.numeric(time_diff)
  f1_mat[4, 1]  = 2/((sum(pred_classes==1 & Ys_test_reduced == 1)/(sum(pred_classes== 1 & Ys_test_reduced == 1)+
                                                                     sum(pred_classes==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(pred_classes==1 & Ys_test_reduced == 1)/(sum(pred_classes==2 & Ys_test_reduced == 1)+
                                                                      sum(pred_classes==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  ###
  #True eigenfunctions & estimated alphas
  ###
  
  st = Sys.time()
  
  fpca_results = get_true_bases(scenario = scenario_funcs, D)
  
  mu_t_hat = fpca_results$mu_hat
  eigen_vals1 = fpca_results$eigen_vals1
  eigen_funcs1 = fpca_results$eigen_funcs1
  eigen_vals2 = fpca_results$eigen_vals2
  eigen_funcs2 = fpca_results$eigen_funcs2
  
  posting_days = (rowSums(X_dat)>1)
  s_mat_hat_train = t(matrix(as.numeric(matrix(posting_days, nrow = J)), nrow = J))
  
  users_to_keep = which(rowSums(s_mat_hat_train>0)!=0)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Ys_train_reduced = Ys_train[users_to_keep]
  scores_train = estimate_scores(X_dat[rows_to_keep,], 
                                 s_mat = s_mat_hat_train[users_to_keep,],
                                 I=length(users_to_keep),  J=J,
                                 eigen_vals1, eigen_vals2,
                                 eigen_funcs1, eigen_funcs2, mu_t_hat)
  users_to_keep_train = users_to_keep
  
  
  users_to_keep = which(rowSums(s_mat_hat_test>0)!=0)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Ys_test_reduced = Ys_test[users_to_keep]
  scores_test = estimate_scores(X_dat_test[rows_to_keep,], 
                                s_mat = s_mat_hat_test[users_to_keep,],
                                I=length(users_to_keep),  J=J,
                                eigen_vals1, eigen_vals2,
                                eigen_funcs1, eigen_funcs2, mu_t_hat)
  users_to_keep_test = users_to_keep
  
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced+1,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  s_mat_hat_test[users_to_keep_test,],
                          s_mat_hat_train =  s_mat_hat_train[users_to_keep_train,])
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  Ys_test_reduced = Ys_test_reduced+1
  #Determine how well the classifier performed
  t1 = table(guess, Ys_test_reduced)
  #record results for proposed method
  misclass_mat[5, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[5,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[5,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[5,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess= 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[5, 1] = as.numeric(time_diff)
  f1_mat[5, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  
  ###
  #True eigenfunctions & true alphas
  ###
  
  alpha_true_l = get_true_alpha(scenario = scenario_S, J=J)
  #alpha_true_l = NA
  
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced+1,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  s_mat_hat_test[users_to_keep_test,],
                          s_mat_hat_train =  s_mat_hat_train[users_to_keep_train,], 
                          alpha_js = alpha_true_l)
  
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  #Ys_test_reduced = Ys_test_reduced+1
  #Determine how well the classifier performed
  t1 = table(guess, Ys_test_reduced)
  #record results for proposed method
  misclass_mat[6, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[6,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[6,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[6,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess= 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[6, 1] = as.numeric(time_diff)
  f1_mat[6, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  ###
  #Latent Curves & Unknown Alpha
  ###
  
  #matplot(fpca_results2$eigen_funcs2%*%diag(c(fpca_results2$eigen_vals2))%*%t(fpca_results2$eigen_funcs2), type="l")
  #matplot(fpca_results2$eigen_funcs1%*%diag(c(fpca_results2$eigen_vals1))%*%t(fpca_results2$eigen_funcs1), type="l")
  
  #set.seed(i)
  #J = 50
  Ys_train = rbinom(N, 1, 0.5)
  Z_tilde = generate_multilevel_data(scenario = scenario_funcs, grid = grid, 
                                     N=N, J=J, binary=F, 
                                     Ys = Ys_train, return_scores_too = T)
  V_scores = t(Z_tilde$V_scores)
  Z_tilde = Z_tilde$Curves_binary
  
  data_ls = generate_data_with_S(scenario = scenario_S, Z_tilde, N, J, Ys_train)
  #All Training & testing sets
  Z_dat = data_ls$Curves_binary
  Z_s_mat = data_ls$s_mat
  
  tt = grid
  #get first and second level eigenfunctions for continuous data
  #Estimates posting per day
  Z_dat_total = t(matrix(rowSums(Z_dat), nrow = J))
  #get p_hat_tilde
  alpha_js = apply(Z_dat_total!=0 , 2, mean)
  
  # #estimate smooth mean function
  z1 = as.vector(colMeans(Z_dat[rowSums(Z_dat)!=0,]))
  gam1 <- gam(z1~s(grid, bs = "cr", m=2, k = 10),
              family="gaussian", method = "REML")
  mu_hat = gam1$fitted.values
  
  #Z_dat[rowSums(Z_dat)!=0,] = Z_dat[rowSums(Z_dat)!=0,] - matrix(rep(mu_hat, each = sum(rowSums(Z_dat)!=0)), ncol=D)
  KZ_Z = cov(Z_dat[rowSums(Z_dat)!=0,])
  p_train = table(Ys_train)/length(Ys_train)
  Ys_rows = rep(Ys_train, each = J)
  
  for(x in 1:length(unique(Ys_train))){
    if(x == 1){
      KZ_Z = cov(Z_dat[rowSums(Z_dat)!=0 & Ys_rows == x-1,])*p_train[x]
    }else{
      KZ_Z = KZ_Z + cov(Z_dat[rowSums(Z_dat)!=0 & Ys_rows == x-1,])*p_train[x]
    }
  }
  
  
  KZ_w = matrix(0, ncol = D, nrow = D)
  counter = 0
  #esitmate first level covariance
  j1_start = 0
  for(i in 1:N){
    j1_star = (i-1)*J+1
    j2_star = (i)*J
    d1 = Z_dat[j1_star:j2_star, ]
    if(sum(rowSums(d1)!=0)>1){
      alpha_cur = mean(rowSums(d1)!=0)
      d1 = d1[rowSums(d1)!=0,]
      #KZ_w = KZ_w+cov(d1)*(alpha_cur*J)/(N)
      KZ_w = KZ_w+cov(d1)/(N)
    }
  }
  KZ_v = KZ_Z-KZ_w
  
  KZ_v = bivariate_smooth(KZ_v, k = 5)
  if(var(c(KZ_w))>0.5){
    KZ_w = bivariate_smooth(KZ_w, k = 5)
  }
  
  fpca_results = estimate_eigenfunctions2(KZ_v, KZ_w,  0.95, 0.95)
  fpca_results$eigen_funcs1 = fpca_results$eigen_funcs1*sqrt(D)
  fpca_results$eigen_vals1 = sqrt(fpca_results$eigen_vals1/D)
  fpca_results$eigen_funcs2 = fpca_results$eigen_funcs2*sqrt(D)
  fpca_results$eigen_vals2 = sqrt(fpca_results$eigen_vals2/D)
  
  mu_t_hat = mu_hat
  eigen_vals1 = fpca_results$eigen_vals1
  eigen_funcs1 = fpca_results$eigen_funcs1
  eigen_vals2 = fpca_results$eigen_vals2
  eigen_funcs2 = fpca_results$eigen_funcs2
  
  users_to_keep = which(rowSums(Z_s_mat>0)!=0)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Ys_train_reduced= Ys_train[users_to_keep]
  scores_train = estimate_scores_normal(Z_dat[rows_to_keep,], 
                                        s_mat = Z_s_mat[users_to_keep,],
                                        I=length(users_to_keep),  J=J,
                                        eigen_vals1, eigen_vals2,
                                        eigen_funcs1, eigen_funcs2, mu_t_hat)
  
  users_to_keep_train = users_to_keep
  
  N_test = 200
  Ys_test = rbinom(N_test, 1, 0.5)
  Z_tilde = generate_multilevel_data(scenario = scenario_funcs, grid = grid, 
                                     N=N_test, J=J, binary=F, 
                                     Ys = Ys_test, return_scores_too = T)
  V_scores = t(Z_tilde$V_scores)
  Z_tilde = Z_tilde$Curves_binary
  
  #ways to generate S's
  data_ls = generate_data_with_S(scenario = scenario_S, Z_tilde, N_test, J, Ys_test)
  
  #All Training & testing sets
  Z_dat_test = data_ls$Curves_binary
  Z_s_mat_test = data_ls$s_mat
  
  users_to_keep = which(rowSums(Z_s_mat_test>0)!=0)
  rows_to_keep = c(sapply(users_to_keep, function(x) ((x-1)*J+1):((x)*J)))
  Ys_test_reduced = Ys_test[users_to_keep]
  scores_test = estimate_scores_normal(Z_dat_test[rows_to_keep,], 
                                       s_mat = Z_s_mat_test[users_to_keep,],
                                       I=length(users_to_keep),  J=J,
                                       eigen_vals1, eigen_vals2,
                                       eigen_funcs1, eigen_funcs2, mu_t_hat)
  users_to_keep_test = users_to_keep
  
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced+1,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  Z_s_mat_test[users_to_keep_test,],
                          s_mat_hat_train =  Z_s_mat[users_to_keep_train,])
  
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  
  #Determine how well the classifier performed
  t1 = table(guess, Ys_test_reduced)
  Ys_test_reduced = Ys_test_reduced+1
  #record results for proposed method
  misclass_mat[7, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[7,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[7,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[7,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess= 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[7, 1] = as.numeric(time_diff)
  f1_mat[7, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1) 
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  
  ###
  #True alpga 
  ###
  
  alpha_true_l = get_true_alpha(scenario = scenario_S, J=J)
  #alpha_true_l = NA
  
  guess = nb_updated_grid(scores = scores_train, classes = Ys_train_reduced+1,
                          prior_g = c(table(Ys_train_reduced)/length(Ys_train_reduced)),
                          scores_test =  scores_test,
                          s_mat_hat_test =  Z_s_mat_test[users_to_keep_test,],
                          s_mat_hat_train =  Z_s_mat[users_to_keep_train,],
                          alpha_js = alpha_true_l)
  
  
  #clock the time
  et = Sys.time()
  time_diff = et - st
  
  #Determine how well the classifier performed
  t1 = table(guess, Ys_test_reduced)
  #Ys_test_reduced = Ys_test_reduced+1
  #record results for proposed method
  misclass_mat[8, 1] = 1- sum(guess==Ys_test_reduced)/length(Ys_test_reduced)
  sens_mat[8,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 1))
  spec_mat[8,1] = sum(guess==2 & Ys_test_reduced == 2)/(sum(guess==2 & Ys_test_reduced == 2)+sum(guess==1 & Ys_test_reduced == 2))
  precision_mat[8,1] = sum(guess==1 & Ys_test_reduced == 1)/(sum(guess= 1 & Ys_test_reduced == 1)+sum(guess==1 & Ys_test_reduced == 2))
  time_mat[8, 1] = as.numeric(time_diff)
  f1_mat[8, 1]  = 2/((sum(guess==1 & Ys_test_reduced == 1)/(sum(guess== 1 & Ys_test_reduced == 1)+
                                                              sum(guess==1 & Ys_test_reduced == 2)))^(-1)
                     +(sum(guess==1 & Ys_test_reduced == 1)/(sum(guess==2 & Ys_test_reduced == 1)+
                                                               sum(guess==1 & Ys_test_reduced == 1)))^(-1))
  
  
  
  #print(i)
  #
  
  return(c(misclass_mat*100, sens_mat*100, spec_mat*100, f1_mat*100, time_mat))
  
  
}








for(N2 in Ns){
  
  #Records the miclassification rates
  misclass_list = list()
  misclass_mat = matrix(NA, nrow = 8,  ncol = ITER)
  
  #Records the times
  time_mat = matrix(NA, nrow = 8,  ncol = ITER)
  time_list = list()
  
  #measures the precision
  precision_mat = matrix(NA, nrow = 8,  ncol = ITER)
  precision_list = list()
  
  #Records the sensitivity
  sens_mat = matrix(NA, nrow = 8,  ncol = ITER)
  sens_list = list()
  
  #Records the specificity
  spec_mat = matrix(NA, nrow = 8,  ncol = ITER)
  spec_list = list()
  
  #measures the precision
  f1_mat = matrix(NA, nrow = 8,  ncol = ITER)
  f1_list = list()
  
  for(j in 1:length(Js)){
    
    J = Js[j]
    #apply core function to each core to parallelize the functions
    results = mclapply(1:ITER, function(x) core_function(x), mc.cores = numCores_to_run)
    results = matrix(unlist(results), ncol = ITER)
    
    #results = sapply(1:ITER, function(x) core_function(x))
    
    #uncomment if you want to see results 
    #print(results)
    
    misclass_mat = results[1:8,]
    time_mat = results[33:40,]
    sens_mat = results[9:16,]
    spec_mat = results[17:24,]
    f1_mat = results[25:32,]
    
    #save results in list
    misclass_list[[j]] = misclass_mat
    time_list[[j]] = time_mat
    sens_list[[j]] = sens_mat
    spec_list[[j]] = spec_mat
    f1_list[[j]] = f1_mat
    
    
  }
  
  
  ####
  # Format Results for latex tables
  ####
  
  ##
  #Misclassification
  ##
  print("Misclassification")
  m1 = lapply(misclass_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(misclass_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    misclass_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    misclass_list.mat = rbind(misclass_list.mat,
                              t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #Times
  ##
  print("Time")
  m1 = lapply(time_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(time_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    time_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    time_list.mat = rbind(time_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #Sensitivity
  ##
  print("Sensitivity")
  m1 = lapply(sens_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(sens_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    sens_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    sens_list.mat = rbind(sens_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  
  ##
  # Specificity
  ##
  print("Specificity")
  m1 = lapply(spec_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(spec_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    spec_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    spec_list.mat = rbind(spec_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), 
                                         " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #F1 results
  ##
  print("F1")
  m1 = lapply(f1_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(f1_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  
  if(N2==Ns[1]){
    f1_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    f1_list.mat = rbind(f1_list.mat,
                        t(matrix(paste(round(unlist(m1), 2), 
                                       " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
}

labels.mat = matrix(c(rep(Ns, each = length(Js)), 
                      rep(Js, length(Ns))), ncol = 2)

#Misclass
misclass_list.mat = cbind(labels.mat,  misclass_list.mat)
print(xtable(misclass_list.mat), include.rownames=FALSE)

misclass_list.mat = misclass_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(misclass_list.mat), include.rownames=FALSE)

#Sens
sens_list.mat = cbind(labels.mat,  sens_list.mat)
print(xtable(sens_list.mat),include.rownames=FALSE)

sens_list.mat = sens_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(sens_list.mat), include.rownames=FALSE)

#spec
spec_list.mat = cbind(labels.mat,  spec_list.mat)
print(xtable(spec_list.mat), include.rownames=FALSE)

spec_list.mat = spec_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(spec_list.mat), include.rownames=FALSE)

#f1
f1_list.mat = cbind(labels.mat,  f1_list.mat)
print(xtable(f1_list.mat), include.rownames=FALSE)

f1_list.mat = f1_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(f1_list.mat), include.rownames=FALSE)









#Set scenario to generate the S data 
scenario_S     = 6



for(N2 in Ns){
  
  #Records the miclassification rates
  misclass_list = list()
  misclass_mat = matrix(NA, nrow = 8,  ncol = ITER)
  
  #Records the times
  time_mat = matrix(NA, nrow = 8,  ncol = ITER)
  time_list = list()
  
  #measures the precision
  precision_mat = matrix(NA, nrow = 8,  ncol = ITER)
  precision_list = list()
  
  #Records the sensitivity
  sens_mat = matrix(NA, nrow = 8,  ncol = ITER)
  sens_list = list()
  
  #Records the specificity
  spec_mat = matrix(NA, nrow = 8,  ncol = ITER)
  spec_list = list()
  
  #measures the precision
  f1_mat = matrix(NA, nrow = 8,  ncol = ITER)
  f1_list = list()
  
  for(j in 1:length(Js)){
    
    J = Js[j]
    #apply core function to each core to parallelize the functions
    results = mclapply(1:ITER, function(x) core_function(x), mc.cores = numCores_to_run)
    results = matrix(unlist(results), ncol = ITER)
    
    #results = sapply(1:ITER, function(x) core_function(x))
    
    #uncomment if you want to see results 
    #print(results)
    
    misclass_mat = results[1:8,]
    time_mat = results[33:40,]
    sens_mat = results[9:16,]
    spec_mat = results[17:24,]
    f1_mat = results[25:32,]
    
    #save results in list
    misclass_list[[j]] = misclass_mat
    time_list[[j]] = time_mat
    sens_list[[j]] = sens_mat
    spec_list[[j]] = spec_mat
    f1_list[[j]] = f1_mat
    
  }
  
  
  
  ####
  # Format Results for latex tables
  ####
  
  ##
  #Misclassification
  ##
  print("Misclassification")
  m1 = lapply(misclass_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(misclass_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    misclass_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    misclass_list.mat = rbind(misclass_list.mat,
                              t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #Times
  ##
  print("Time")
  m1 = lapply(time_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(time_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    time_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    time_list.mat = rbind(time_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #Sensitivity
  ##
  print("Sensitivity")
  m1 = lapply(sens_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(sens_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    sens_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    sens_list.mat = rbind(sens_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  
  ##
  # Specificity
  ##
  print("Specificity")
  m1 = lapply(spec_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(spec_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    spec_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    spec_list.mat = rbind(spec_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), 
                                         " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #F1 results
  ##
  print("F1")
  m1 = lapply(f1_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(f1_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  
  if(N2==Ns[1]){
    f1_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    f1_list.mat = rbind(f1_list.mat,
                        t(matrix(paste(round(unlist(m1), 2), 
                                       " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
}

labels.mat = matrix(c(rep(Ns, each = length(Js)), 
                      rep(Js, length(Ns))), ncol = 2)

#Misclass
misclass_list.mat = cbind(labels.mat,  misclass_list.mat)
print(xtable(misclass_list.mat), include.rownames=FALSE)

misclass_list.mat = misclass_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(misclass_list.mat), include.rownames=FALSE)

#Sens
sens_list.mat = cbind(labels.mat,  sens_list.mat)
print(xtable(sens_list.mat),include.rownames=FALSE)

sens_list.mat = sens_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(sens_list.mat), include.rownames=FALSE)

#spec
spec_list.mat = cbind(labels.mat,  spec_list.mat)
print(xtable(spec_list.mat), include.rownames=FALSE)

spec_list.mat = spec_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(spec_list.mat), include.rownames=FALSE)

#f1
f1_list.mat = cbind(labels.mat,  f1_list.mat)
print(xtable(f1_list.mat), include.rownames=FALSE)

f1_list.mat = f1_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(f1_list.mat), include.rownames=FALSE)






#Set scenario to generate the S data 
scenario_S   = 7



for(N2 in Ns){
  
  #Records the miclassification rates
  misclass_list = list()
  misclass_mat = matrix(NA, nrow = 8,  ncol = ITER)
  
  #Records the times
  time_mat = matrix(NA, nrow = 8,  ncol = ITER)
  time_list = list()
  
  #measures the precision
  precision_mat = matrix(NA, nrow = 8,  ncol = ITER)
  precision_list = list()
  
  #Records the sensitivity
  sens_mat = matrix(NA, nrow = 8,  ncol = ITER)
  sens_list = list()
  
  #Records the specificity
  spec_mat = matrix(NA, nrow = 8,  ncol = ITER)
  spec_list = list()
  
  #measures the precision
  f1_mat = matrix(NA, nrow = 8,  ncol = ITER)
  f1_list = list()
  
  for(j in 1:length(Js)){
    
    J = Js[j]
    #apply core function to each core to parallelize the functions
    results = mclapply(1:ITER, function(x) core_function(x), mc.cores = numCores_to_run)
    results = matrix(unlist(results), ncol = ITER)
    
    #results = sapply(1:ITER, function(x) core_function(x))
    
    #uncomment if you want to see results 
    #print(results)
    
    misclass_mat = results[1:8,]
    time_mat = results[33:40,]
    sens_mat = results[9:16,]
    spec_mat = results[17:24,]
    f1_mat = results[25:32,]
    
    #save results in list
    misclass_list[[j]] = misclass_mat
    time_list[[j]] = time_mat
    sens_list[[j]] = sens_mat
    spec_list[[j]] = spec_mat
    f1_list[[j]] = f1_mat
    
  }
  
  
  
  ####
  # Format Results for latex tables
  ####
  
  ##
  #Misclassification
  ##
  print("Misclassification")
  m1 = lapply(misclass_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(misclass_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    misclass_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    misclass_list.mat = rbind(misclass_list.mat,
                              t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #Times
  ##
  print("Time")
  m1 = lapply(time_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(time_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    time_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    time_list.mat = rbind(time_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #Sensitivity
  ##
  print("Sensitivity")
  m1 = lapply(sens_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(sens_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    sens_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    sens_list.mat = rbind(sens_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  
  ##
  # Specificity
  ##
  print("Specificity")
  m1 = lapply(spec_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(spec_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  if(N2==Ns[1]){
    spec_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    spec_list.mat = rbind(spec_list.mat,
                          t(matrix(paste(round(unlist(m1), 2), 
                                         " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
  ##
  #F1 results
  ##
  print("F1")
  m1 = lapply(f1_list, function(x) rowMeans(x, na.rm=T))
  s1 = lapply(f1_list, function(x) apply(x, 1, function(y) sd(y, na.rm=T)/sqrt(ITER)))
  print(xtable(t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))))
  
  
  if(N2==Ns[1]){
    f1_list.mat = t(matrix(paste(round(unlist(m1), 2), " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8))
  }else{
    f1_list.mat = rbind(f1_list.mat,
                        t(matrix(paste(round(unlist(m1), 2), 
                                       " (", round(unlist(s1), 2), ")", sep = ""), nrow = 8)))
  }
  
}

labels.mat = matrix(c(rep(Ns, each = length(Js)), 
                      rep(Js, length(Ns))), ncol = 2)

#Misclass
misclass_list.mat = cbind(labels.mat,  misclass_list.mat)
print(xtable(misclass_list.mat), include.rownames=FALSE)

misclass_list.mat = misclass_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(misclass_list.mat), include.rownames=FALSE)

#Sens
sens_list.mat = cbind(labels.mat,  sens_list.mat)
print(xtable(sens_list.mat),include.rownames=FALSE)

sens_list.mat = sens_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(sens_list.mat), include.rownames=FALSE)

#spec
spec_list.mat = cbind(labels.mat,  spec_list.mat)
print(xtable(spec_list.mat), include.rownames=FALSE)

spec_list.mat = spec_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(spec_list.mat), include.rownames=FALSE)

#f1
f1_list.mat = cbind(labels.mat,  f1_list.mat)
print(xtable(f1_list.mat), include.rownames=FALSE)

f1_list.mat = f1_list.mat[,c(1,2,3,4,6,7,9)]
print(xtable(f1_list.mat), include.rownames=FALSE)


