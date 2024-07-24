##################
#Functions for "Classification of Social Media Accounts Using a Generalized Multilevel Functional Analysis"
#
#Authors: Anthony Weishampel, Ana-Maria Staicu, Bill Rand
#Date: 7/16/2024
#
##################

#dependencies to run all of the code
library(parallel)
library(xtable)
library(fda)
library(refund)
library(Matrix)
library(MASS)
library(arm)
library(mgcv)
library(readr)

library(randomForest)
library(splines)
library(fields)
library(class)
library(gamm4)
library(lme4)
library(xtable)

############
# Functions for analyses
############

#' Functions used to generate data in the various scenarios in accompanying paper
#' @name generate_multilevel_data
#' @param scenario Which scenario to generate data for: 1=A, 2=B, 3=C In paper
#' @param grid which set of points within [0,1] will data be observed on
#' @param N Total number of subjects to generate data for
#' @param p vector showing distribution of the two groups
#' @param J number of realizations
#' @param sigma variance in the latent curves
#' @param binary Reports whether to return binary or latent curves
#' @param Ys Groups of the individuals
#' @param return_scores_too T/F values to determine whether to return score coefficients too
#' @return: Binary or latent curves
generate_multilevel_data <- function(scenario = 1 , grid=seq(from = 0,to = 1, length.out = 48),
                                     N, p = rep(0.5, 2), J = 7,
                                     sigma = 0, binary = T,
                                     Ys = rbinom(N, 1, 0.5),
                                     return_scores_too = F){

  #For Scenario A in the paper
  if(scenario == 1){

    #number of outputs
    D = length(grid)

    #Number of subjects per groups
    # N1 = round(p[1]*N)
    # N2 = N-N1

    I=N
    D = length(grid)  ## size of grid for observations
    IJ = I * J                        ## total number of observed functions

    #true mean
    #beta0_true = sin(2*pi*grid)+cos(2*pi*grid)
    beta0_true = -0.5-sin(2*pi*grid)-cos(2*pi*grid)
    #beta0_true = 0*grid

    #set variance struture or scneario A
    theta1<- theta0 <-1/(c(1:40)^2)
    theta1[c(1,9:40)] = 0
    theta0[c(1,9:40)] = 0
    #set mean functions
    mu0 <-  c(0, -0.5, 1, -0.5,1,-0.5)
    mu1 <- c(0, -0.75, 0.75, -0.15,1.4,0.1)

    #mu0 <-  c(0, 0, 0, 0, 0, 0)
    #mu1 <- c(0, 0, 0, 0, 0, 0)


    #generate latent functions for both groups given the mean and variance
    # V0 <- t(generate_latent_process(theta=theta0, mu_coeff = mu0, tt=seq(0,1, len=D), n=N1))
    # V1 <- t(generate_latent_process(theta=theta1, mu_coeff = mu1, tt=seq(0,1, len=D), n=N2))

    test1 = function(x, return_scores_too){
      if(Ys[x]==0){
        return(generate_latent_process(theta=theta0, mu_coeff =mu0, tt=seq(0,1, len=D), n=1, return_scores_too))
      }
      if(Ys[x]==1){
        return(generate_latent_process(theta=theta1, mu_coeff =mu1, tt=seq(0,1, len=D), n=1, return_scores_too))
      }
    }

    vec = matrix(1:N, ncol = 1)
    if(return_scores_too){
      Vall = apply(vec, 1,  function(x) test1(x, return_scores_too))
      V0 = t(matrix(unlist(lapply(Vall, function(x) x$X)), nrow = D))
      V_scores = matrix(unlist(lapply(Vall, function(x) x$scores)), ncol = N)
    }else{
      V0 = t(apply(vec, 1,  function(x) test1(x, return_scores_too)))
    }


    #Second Level eigenfunctions
    psi_21 = sqrt(2)*cos(4*pi*grid)
    psi_22 = sqrt(2)*sin(4*pi*grid)
    psi_2=cbind(psi_21, psi_22)

    lambda2=matrix(c(0.25, 0.08), nrow=1)

    #select second level eigenfunctions
    c_21=rnorm(IJ, 0, lambda2[1])
    c_22=rnorm(IJ, 0, lambda2[2])

    #convert values into matrices at the observed values
    beta0_true_mat = kronecker(matrix(1, nrow=IJ), matrix(beta0_true, ncol=D)) #Mean function

    for(i in 1:N){
      x = V0[i,]
      if(i == 1){
        V_mat = matrix(rep(x, each = J),nrow=J)
      }else{
        V_mat = rbind(V_mat, matrix(rep(x, each = J),nrow=J))
      }

    }

    #second level/daily effects
    c_21_mat = kronecker( matrix(c_21, ncol=1), matrix(psi_21, nrow=1))
    c_22_mat = kronecker( matrix(c_22, ncol=1), matrix(psi_22, nrow=1))

    Curves = beta0_true_mat + V_mat +
      c_21_mat + c_22_mat

    #if binary curves wanted
    if(binary){
      #binary_values
      Curves_binary <- t(generate_binary_fns(t(Curves)))
    }else{
      #non_binary
      epsilon = matrix(rnorm(N*D*J, 0, sigma), ncol=D)
      Curves_binary = Curves+epsilon
    }
    if(return_scores_too){

      return(list(Curves_binary = Curves_binary, V_scores=V_scores))

    }
    return(Curves_binary)

  }
  if(scenario == 2){

    #number of outputs
    D = length(grid)

    #Number of subjects per groups
    # N1 = round(p[1]*N)
    # N2 = N-N1

    I=N
    D = length(grid)  ## size of grid for observations
    IJ = I * J                        ## total number of observed functions

    #true mean
    #beta0_true = sin(2*pi*grid)+cos(2*pi*grid)
    beta0_true = -0.5-sin(2*pi*grid)-cos(2*pi*grid)
    #beta0_true = 0*grid

    #set variance struture or scneario A
    theta1<- theta0 <-1/(c(1:40)^2)
    theta1[c(1,9:40)] = 0
    theta0[c(1,9:40)] = 0
    #set mean functions
    mu0 <-  c(0, -0.5, 1, -0.5,1,-0.5)
    mu1 <- c(0, -0.75, 0.75, -0.15,1.4,0.1)

    mu0 <-  -1*theta1[1:8]/2
    mu1 <- theta0[1:8]/2

    test1 = function(x, return_scores_too){
      if(Ys[x]==0){
        return(generate_latent_process(theta=theta0, mu_coeff =mu0, tt=seq(0,1, len=D), n=1, return_scores_too))
      }
      if(Ys[x]==1){
        return(generate_latent_process(theta=theta1, mu_coeff =mu1, tt=seq(0,1, len=D), n=1, return_scores_too))
      }
    }

    vec = matrix(1:N, ncol = 1)
    if(return_scores_too){
      Vall = apply(vec, 1,  function(x) test1(x, return_scores_too))
      V0 = t(matrix(unlist(lapply(Vall, function(x) x$X)), nrow = D))
      V_scores = matrix(unlist(lapply(Vall, function(x) x$scores)), ncol = N)
    }else{
      V0 = t(apply(vec, 1,  function(x) test1(x, return_scores_too)))
    }

    lambda2=matrix(c(0.25), nrow=1)

    #select second level eigenfunctions
    W_IJ=rnorm(IJ, 0, lambda2[1])

    #convert values into matrices at the observed values
    beta0_true_mat = kronecker(matrix(1, nrow=IJ), matrix(beta0_true, ncol=D)) #Mean function

    for(i in 1:N){
      x = V0[i,]
      if(i == 1){
        V_mat = matrix(rep(x, each = J),nrow=J)
      }else{
        V_mat = rbind(V_mat, matrix(rep(x, each = J),nrow=J))
      }

    }

    #second level/daily effects
    W_mat = t(matrix(rep(W_IJ, each = D), nrow = D))

    Curves = beta0_true_mat + V_mat + W_mat

    #if binary curves wanted
    if(binary){
      #binary_values
      Curves_binary <- t(generate_binary_fns(t(Curves)))
    }else{
      #non_binary
      epsilon = matrix(rnorm(N*D*J, 0, sigma), ncol=D)
      Curves_binary = Curves+epsilon
    }
    if(return_scores_too){

      return(list(Curves_binary = Curves_binary, V_scores=V_scores))

    }
    return(Curves_binary)

  }
  if(scenario == 3){

    #number of outputs
    D = length(grid)

    #Number of subjects per groups
    # N1 = round(p[1]*N)
    # N2 = N-N1

    I=N
    D = length(grid)  ## size of grid for observations
    IJ = I * J                        ## total number of observed functions

    #true mean
    #beta0_true = sin(2*pi*grid)+cos(2*pi*grid)
    beta0_true = -0.5-sin(2*pi*grid)-cos(2*pi*grid)
    #beta0_true = 0*grid

    #set variance struture or scneario A
    theta0 <- exp(-c(1:40)/3)
    theta1 <- exp(-c(1:40)/2)
    theta1[c(1,9:40)] = 0
    theta0[c(1,9:40)] = 0
    #set mean functions

    mu0 <-  rep(0, 8)
    mu1 <-  rep(0, 8)

    test1 = function(x, return_scores_too){
      if(Ys[x]==0){
        return(generate_latent_process(theta=theta0, mu_coeff =mu0, tt=seq(0,1, len=D), n=1, return_scores_too))
      }
      if(Ys[x]==1){
        return(generate_latent_process(theta=theta1, mu_coeff =mu1, tt=seq(0,1, len=D), n=1, return_scores_too))
      }
    }

    vec = matrix(1:N, ncol = 1)
    if(return_scores_too){
      Vall = apply(vec, 1,  function(x) test1(x, return_scores_too))
      V0 = t(matrix(unlist(lapply(Vall, function(x) x$X)), nrow = D))
      V_scores = matrix(unlist(lapply(Vall, function(x) x$scores)), ncol = N)
    }else{
      V0 = t(apply(vec, 1,  function(x) test1(x, return_scores_too)))
    }

    #convert values into matrices at the observed values
    beta0_true_mat = kronecker(matrix(1, nrow=IJ), matrix(beta0_true, ncol=D)) #Mean function

    for(i in 1:N){
      x = V0[i,]
      if(i == 1){
        V_mat = matrix(rep(x, each = J),nrow=J)
      }else{
        V_mat = rbind(V_mat, matrix(rep(x, each = J),nrow=J))
      }

    }


    #Second Level eigenfunctions
    psi_21 = sqrt(2)*cos(4*pi*grid)
    psi_22 = sqrt(2)*sin(4*pi*grid)
    psi_2=cbind(psi_21, psi_22)

    lambda2=matrix(c(0.25, 0.08), nrow=1)

    #select second level eigenfunctions
    c_21=rnorm(IJ, 0, lambda2[1])
    c_22=rnorm(IJ, 0, lambda2[2])

    #second level/daily effects
    c_21_mat = kronecker( matrix(c_21, ncol=1), matrix(psi_21, nrow=1))
    c_22_mat = kronecker( matrix(c_22, ncol=1), matrix(psi_22, nrow=1))

    Curves = beta0_true_mat + V_mat +
      c_21_mat + c_22_mat

    #if binary curves wanted
    if(binary){
      #binary_values
      Curves_binary <- t(generate_binary_fns(t(Curves)))
    }else{
      #non_binary
      epsilon = matrix(rnorm(N*D*J, 0, sigma), ncol=D)
      Curves_binary = Curves+epsilon
    }
    if(return_scores_too){

      return(list(Curves_binary = Curves_binary, V_scores=V_scores))

    }
    return(Curves_binary)

  }
}


#' Functions used to generate latent state data in the various scenarios in accompanying paper
#' @param scenario Which scenario to generate S data
#' @param Curves_binary The binary-valued functional data
#' @param N Total number of subjects to generate data for
#' @param J Number of realizations
#' @param Ys Group assignments for the N individuals
#' @param alpha1 probability for latent state i ngroup 1
#' @param alpha2 probability for latent state in group 2
#' @return Binary or latent curves with latent states
generate_data_with_S = function(scenario = 3, Curves_binary, N, J, Ys, alpha1 = 2/3, alpha2 = 1/4){

  #define the S_matrix
  S_mat = matrix(NA, nrow = N, ncol = J)

  #constant
  if(scenario == 1){

    alpha1 = 0.4
    alpha2 = 0.6

    test1 = function(x){
      if(x==1){
        return(rbinom(J, 1, alpha1))
      }
      return(rbinom(J, 1, alpha2))
    }

    S_mat = t(sapply(Ys, test1))

  }
  if(scenario == 2){

    alpha1 = 0.5
    alpha2 = 0.25

    #Make sure to change the alpha1 scores
    test1 = function(x){
      if(x==1){
        return(c(rbinom(floor(J/2), 1, alpha1), rbinom(J-floor(J/2), 1, alpha2)))
      }else{
        probs = rep(c(alpha1, alpha2), ceiling(J/2))
        return(rbinom(J, 1, probs))
      }
    }

    S_mat = t(sapply(Ys, test1))

  }
  if(scenario == 3){

    #Make sure to change the alpha1 scores
    test1 = function(x){
      if(x==1){
        return(rbinom(J, 1, alpha1))
      }else{
        Js = 1:J
        probs = 2/3-(1/Js)*(1/4)
        return(rbinom(J, 1, probs))
      }
    }

    S_mat = t(sapply(Ys, test1))

  }
  if(scenario == 5){

    alpha1 = 0.5
    alpha2 = 0.5

    test1 = function(x){
      if(x==1){
        return(rbinom(J, 1, alpha1))
      }
      return(rbinom(J, 1, alpha2))
    }

    S_mat = t(sapply(Ys, test1))

  }
  if(scenario == 4){

    alpha1 = c(0.4543210 , 0.4098765 , 0.4370370 , 0.4197531 , 0.3975309 , 0.4345679 , 0.4246914 , 0.4222222,
               0.3851852 , 0.3925926 , 0.4543210 , 0.4493827 , 0.4543210 , 0.4469136 , 0.4098765 , 0.3950617,
               0.4024691 , 0.4493827 , 0.4617284 , 0.4814815 , 0.4296296 , 0.4518519 , 0.4395062 , 0.4617284,
               0.4395062 , 0.4666667 , 0.4567901 , 0.4716049 , 0.4320988 , 0.4592593 , 0.4592593 , 0.4790123,
               0.4938272 , 0.5037037 , 0.5037037 , 0.4888889 , 0.4691358 , 0.4617284 , 0.4740741 , 0.5358025,
               0.5358025 , 0.5209877 , 0.5135802 , 0.5012346 , 0.5234568 , 0.5185185 , 0.5111111 , 0.5555556,
               0.5407407 , 0.5407407)
    alpha2 = c(0.26824458 , 0.36982249 , 0.08974359 , 0.36883629 , 0.11143984 , 0.40729783 , 0.09861933 , 0.56213018,
               0.37080868 , 0.17751479 , 0.06213018 , 0.13214990 , 0.08678501 , 0.09566075 , 0.19526627 , 0.16370809,
               0.26331361 , 0.35305720 , 0.23372781 , 0.26923077 , 0.37771203 , 0.39743590 , 0.18639053 , 0.20216963,
               0.14003945 , 0.29881657 , 0.16863905 , 0.18441815 , 0.25345168 , 0.32741617 , 0.21794872 , 0.17357002,
               0.06607495 , 0.14990138 , 0.16962525 , 0.27514793 , 0.10946746 , 0.24260355 , 0.20019724 , 0.18244576,
               0.26725838 , 0.23076923 , 0.24358974 , 0.19428008 , 0.22879684 , 0.15976331 , 0.32248521 , 0.18836292,
               0.24161736 , 0.52761341)

    #Make sure to change the alpha1 scores
    test1 = function(x){
      if(x==1){
        return(rbinom(J, 1, alpha1))
      }else{
        return(rbinom(J, 1, alpha2))
      }
    }

    alpha1 = alpha1[1:J]
    alpha2 = alpha2[1:J]

    S_mat = t(sapply(Ys, test1))

  }
  if(scenario == 6){

    beta1 = c(-1.13, 1.15, 0.81, 0.64)
    beta2 = c(-1.22, 0.63, 0.66, 0.04)
    pi1 = c(0.20, 0.08, 0.07, 0.09, 0.08, 0.08, 0.09, 0.31)
    pi2 = c(0.41, 0.11, 0.11, 0.06, 0.11, 0.06, 0.06, 0.08)
    initial_combos = as.matrix.data.frame(expand.grid(rep(list(0:1),3)))


    #Make sure to change the alpha1 scores
    test1 = function(x){
      if(x==1){
        S_s = rep(NA, J)
        initals = c(initial_combos[sample(1:8, 1, prob = pi1, replace = T),])
        S_s[1:3] = initals
        for(j in 4:J){
          S_pre =  S_s[(j-3):(j-1)]
          alpha.cur = invlogit(beta1[1] + sum(beta1[2:4]*S_pre))
          S_s[j] = rbinom(1, 1, alpha.cur)
        }
        return(c(S_s))
      }else{
        S_s = rep(NA, J)
        initals = c(initial_combos[sample(1:8, 1, prob = pi2, replace = T),])
        S_s[1:3] = initals
        for(j in 4:J){
          S_pre =  S_s[(j-3):(j-1)]
          alpha.cur = invlogit(beta2[1] + sum(beta2[2:4]*S_pre))
          S_s[j] = rbinom(1, 1, alpha.cur)
        }
        return(c(S_s))
      }
    }

    S_mat = t(sapply(Ys, test1))

  }


  if(scenario == 7){

    beta1 = c(-1.13, 1.15, 0.81, 0.64)
    beta2 = c(-1.22, 0.63, 0, 0)
    pi1 = c(0.20, 0.08, 0.07, 0.09, 0.08, 0.08, 0.09, 0.31)
    pi2 = c(0.41, 0.11, 0.11, 0.06, 0.11, 0.06, 0.06, 0.08)
    initial_combos = as.matrix.data.frame(expand.grid(rep(list(0:1),3)))


    #Make sure to change the alpha1 scores
    test1 = function(x){
      if(x==1){
        S_s = rep(NA, J)
        initals = c(initial_combos[sample(1:8, 1, prob = pi1, replace = T),])
        S_s[1:3] = initals
        for(j in 4:J){
          S_pre =  S_s[(j-3):(j-1)]
          alpha.cur = invlogit(beta1[1] + sum(beta1[2:4]*S_pre))
          S_s[j] = rbinom(1, 1, alpha.cur)
        }
        return(c(S_s))
      }else{
        S_s = rep(NA, J)
        initals = c(initial_combos[sample(1:8, 1, prob = pi2, replace = T),])
        S_s[1:3] = initals
        for(j in 4:J){
          S_pre =  S_s[(j-3):(j-1)]
          alpha.cur = invlogit(beta2[1] + sum(beta2[2:4]*S_pre))
          S_s[j] = rbinom(1, 1, alpha.cur)
        }
        return(c(S_s))
      }
    }

    S_mat = t(sapply(Ys, test1))

  }



  return_ls = list()
  return_ls$s_mat = S_mat
  S_mat_s = c(t(S_mat))
  D = dim(Curves_binary)[2]
  test1 = function(x){
    if(S_mat_s[x]==0){
      return(rep(0, D))
    }
    return(Curves_binary[x,])
  }

  NJ = dim(Curves_binary)[1]
  return_ls$Curves_binary = t(sapply(1:NJ, test1))

  return(return_ls)

}



#' Function that makes a matrix positive semi definite
#' @param x matrix to make semi-positive definite
#' @return  semi-positive definite of x
make_pos_semi_def = function(x){
  x2 = svd(x)
  #remove negative vals
  x2$d=ifelse(x2$d>0, x2$d, 0)
  #remake matrix
  x2=x2$u%*%diag(x2$d)%*%t(x2$v)
  return(x2)
}


#' Function: Get number of functions based on the pve and eigenvalues
#' @param pve Value [0,1] to determine number of eigenfunctions
#' @param vec vector of eigenvalues
#' @param set_max_number If you want a maximum number of values
#' @return The number of eigenfunctions and eigenvalues for KL-approximation
get_length_pve = function(pve, vec, set_max_number = NA){

  #total sum of eigenvals
  s=sum(vec)
  vec2 = cumsum(vec)
  vec=vec2/s
  #return(which(vec>=pve)[1])
  if(!is.na(set_max_number)){
    if(which(vec>=pve)[1] < set_max_number){
      return(which(vec>=pve)[1])
    }else{
      return(set_max_number)
    }
  }
  return(which(vec>=pve)[1])
}



#' Function to estimate eigenfunctions
#' @param K_b estimate of a covariance matrix
#' @param pve Proportion of variance explained
#' @param fix_num_of_functions set the number of eigenfunctions to be returned
#' @return The eigenvalues and eigenvectors from
estimate_eigenfunctions = function(K_b, pve=0.98, fix_num_of_functions=NA){

  #SVD on matrix
  svd_kb = svd(K_b)

  #Only get the pve length if the number of functions is not given
  if(is.na(fix_num_of_functions)){
    #get number of components
    pb = get_length_pve(pve, svd_kb$d)
  }else{
    pb = fix_num_of_functions
  }

  eigen_vals1 = svd_kb$d[1:pb]
  eigen_funcs1 = svd_kb$v[,1:pb]

  return(list(eigen_vals1=eigen_vals1, eigen_funcs1=eigen_funcs1))

}


#' Function used to generate the latent curves for simulation scenarios where I define various scenarios, inspired by Delaigle and Hall(2012)
#' @param theta  Define latent process X_i using Fourier basis
#' @param mu_coeff vector of mean coefficients
#' @param tt grid of points which the functions are observed
#' @param n number of individuals to generate data for
#' @param return_scores_too Indicator to determine whether or not to return the coefficients too
#' @return latent process observed on points defined by grid
generate_latent_process <- function(theta, mu_coeff, tt, n=20, return_scores_too = F){
  # tt<- seq(0,1, len=101)
  K <- length(theta); Khalf <- round(K/2)
  Kseqeven <- 2*(1:Khalf); Kseqodd<- Kseqeven-1

  Phitt1 <- sapply(c(1:Khalf), function(j) sqrt(2) * sin(2*pi*j*tt))
  Phitt2 <- sapply(c(1:Khalf), function(j) sqrt(2) * cos(2*pi*j*tt))
  Phitt <- matrix(0, ncol=2*Khalf, nrow=length(tt))
  Phitt[,Kseqeven] <- Phitt1
  Phitt[,Kseqodd] <- Phitt2
  Phitt <-cbind(rep(1, length(tt)), Phitt)[, 1:K]

  Z <- matrix(rnorm (K*n), ncol=n)
  Lambda_half <- diag(sqrt(theta))

  mu <- rep(0, K)
  mu[1:length(mu_coeff)] <- mu_coeff
  X <- Phitt %*% (Lambda_half%*% Z + mu)
  if(return_scores_too){
    return(list(X = X, scores = (Lambda_half%*% Z+mu)[theta!=0]))
  }else{
    return(X)
  }
  X
}



#' Function used to generate the binary valued functional data from latent curves
#' @param X is the latent functional data
#' @return Binary valued functional data determined by X
generate_binary_fns <- function(X){
  length_tt <- nrow(X)
  probs<-  1/(1+exp(-X))
  out <- matrix(rbinom(length(X), 1,  c(probs)), nrow=length_tt)
  out
}


#' Function to return the logit of x
#' @param x value to input
#' @return Returns the logit of x
logit <- function(x){
  return(log(x/(1-x)))
}

#' Function to return the inverse of the logit
#' @param x value to input
#' @return Returns the invlogit of x
invlogit <- function(x){
  return(1/(1+exp(-x)))
}

#' function to return the derivative of the logit
#' @param x value to input
#' @return Returns the derivative of the logit of x
d_logit <- function(x){
  return( invlogit(x)*(1-invlogit(x)))
}




#' Function: To get the density values of new scores for an individual in a given class
#' @param densities List of densities of scores within each group
#' @param scores matrix of scores for the N individuals
#' @param classes matrix of classes (these are not the true classes but the value to be evaluated)
#' @param i which individual to evaluate
#' @return Density values of the new scores for individual i
get_pdf_den2 = function(densities, scores, classes, i){

  #get score for current user
  score_cur = scores[i,]
  #get class for current user
  class_cur = classes[i]
  #get density for current class
  dens_cur = densities[[class_cur]]

  pdf.vals = rep(NA, length(dens_cur))
  #for each component
  for(k in 1:length(dens_cur)){

    #get K comp density
    dens_cur_k = dens_cur[[k]]

    #approximate the function
    approx_fun_den = approxfun(dens_cur_k)

    #get new score
    xnew = score_cur[k]
    #get value in the density
    #pdf.vals[k]  = approx_fun_den(xnew)
    #changed to probability approx
    delta0 = diff(dens_cur_k$x)[1]

    lower0 = xnew
    upper0 = xnew+delta0

    #if new value is outside of defined range
    if(xnew<=min(dens_cur_k$x)){
      xnew=min(dens_cur_k$x)
    }
    #if new vlaue is outside of defined range
    if(xnew>=max(dens_cur_k$x)){
      xnew=max(dens_cur_k$x)
    }
    
    #get value in the density
    pdf.vals[k]  = approx_fun_den(xnew)
  
  }
  return(pdf.vals)
}

#' Function: Estimate predicted values for a given value of scaling parameter h
#' @param scores N x K matrix of scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param h multiplier for kernel based density function
#' @param s_mat_hat_train matrix of estimated S latent state values training set
#' @param s_mat_hat_test matrix of estimated S latent state values testing set
#' @param P_max Lag in the gAR models
#' @param static_train Covariates for the classifier in the training set accounts
#' @param static_test Covariates for the classifier in the testing set accounts
#' @param alpha_js Estimated probability of latent states across the realizations
#' @return Predicted groups for the individuals
nb_updated = function(scores, classes, prior_g, scores_test,
                      s_mat_hat_train, s_mat_hat_test, alpha_js=NA,  h = 1.06, P_max = 3,
                      static_train = NA, static_test = NA){

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  #get K
  nd = dim(scores)[2]
  #get number of groups
  ng = length(unique(classes))
  #initialize list
  densities = list()
  #estimate density at each K for each group
  for(i in 1:ng){
    densities[[i]]=list()
    for(k in 1:nd){
      densities[[i]][[k]] = density(scores[classes==i,k],
                                    kernel = "gaussian",
                                    bw = h*sd(scores[classes==i,k]))
    }
  }

  # get number of test functions
  n_test = dim(scores_test)[1]
  p.mat = matrix(NA, nrow = n_test, ncol = length(prior_g))
  vec = matrix(1:n_test, ncol = 1)

  #get_ps = rep(NA, ng)
  #P_max = min(P_max, floor(J/5))
  #tol = 0.001
  #for(l in 1:ng){
  #  aics.cur = sapply(0:P_max, function(x) fit_ajs_model(l, x, s_mat_hat_train, aic = T, classes, static_train))
  #  get_ps[l] = (0:P_max)[which.min(aics.cur)]
  #  #diff(aics.cur)/aics.cur[2:P_max]
  #}

  #set ps to 3
  #get_ps = rep(P_max, ng)
  get_ps = rep(3, ng)
  
  models_ls = list()
  p.mat_s = matrix(NA, nrow = n_test, ncol = length(unique(classes)))

  for(l in 1:ng){

    models_ls[[l]] = fit_ajs_model(l, get_ps[l], s_mat_hat_train, classes = classes, static_train = static_train)
    p.mat_s[,l]  = sapply(1:n_test, function(x) predict_new_ind_group_model(x, l, get_ps[l],  models_ls, s_mat_hat_test, static_test))

  }

  #For each group get the Bayes classifier value of probability of being in that group
  for(i in 1:ng){
    #apply for each user get the probability for each component in that group
    pdf.vals.test.1 = t(apply(vec, 1,
                              function(x)  get_pdf_den2(densities, scores_test, rep(i, n_test), x)))
    #apply for each user to get the product of those K probabilities
    pdf_vals = apply(pdf.vals.test.1, 1, prod)
    pdf.vals.test.1 = cbind.data.frame(pdf_vals, p.mat_s[,i])
    pdf_vals = apply(pdf.vals.test.1, 1, prod)
    #pdf_vals = apply(pdf.vals.test.1, 1, mean)
    #pdf_vals = apply(pdf.vals.test.1, 1, function(x) sum(x*c(gamma0, 1-gamma0)))
    #multiply by prior probability
    p.mat[,i] = prior_g[i] * pdf_vals #p.mat is now matrix of Bayes probabilities
  }

  #get guess based on which is the max
  guess = apply(p.mat, 1, which.max)

  return(guess)

}



#' Function: Grid search to estimate predicted values and estimate the values of h
#' @param scores N x K matrix of 1st level scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param min.h min possible value for the multiplier
#' @param max.h maximum possible value for the multiplier
#' @param n_grid number of values between min.h and max.h to search over
#' @param CV Number of folds for cross validation
#' @param return_h T/F to return the value of the multiplier
#' @param return_prob T/F to return group Bayes classifier probability for each individual in testing set
#' @param static_train Covariates for the classifier in the training set accounts
#' @param static_test Covariates for the classifier in the testing set accounts
#' @param alpha_js Estimated probability of latent states across the realizations
#' @param s_mat_hat_train Latent state matrix of estimated values for the training set accounts
#' @param s_mat_hat_test Latent state matrix of estimated values for the testing set accounts
#' @param P_max Lag of the gAR models
#' @return Predicted group for each individual
nb_updated_grid = function(scores, s_mat_hat_train, classes,
                           prior_g, scores_test, s_mat_hat_test, alpha_js = NA,
                           min.h = 0.4, max.h = 1.6,
                           CV = 5, n_grid = 5, return_h = F, return_prob = F, P_max = 5,
                           static_train = NA, static_test = NA){

  #create matrix for apply functions
  vec.cv = matrix(1:CV, ncol = 1)


  #create vector of possible CV groups to each account
  #add extra incase unequal distribution of groups
  cvgroups = rep(1:CV, (length(classes)/CV+1))
  #remove the unneeded values
  cvgroups = cvgroups[1:length(classes)]
  #randomly assign CV group to each account
  cvgroups = sample(cvgroups, size = length(classes), replace = F)

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    #If k == 1 change to vector
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  # define function here to use the cvgroups this function
  # will be used in the following apply statement
  get.cv.h.static = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated(scores[cvgroups!=x,], classes[cvgroups!=x],
                     c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                     scores[cvgroups==x,],
                     s_mat_hat_train = s_mat_hat_train[cvgroups!=x,],
                     s_mat_hat_test = s_mat_hat_train[cvgroups==x,],
                     h = h.val, alpha_js = alpha_js, P_max = P_max,
                     static_train = static_train[cvgroups!=x, ],
                     static_test = static_train[cvgroups==x, ]) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }
  get.cv.h = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated(scores[cvgroups!=x,], classes[cvgroups!=x],
                     c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                     scores[cvgroups==x,],
                     s_mat_hat_train = s_mat_hat_train[cvgroups!=x,],
                     s_mat_hat_test = s_mat_hat_train[cvgroups==x,],
                     h = h.val, alpha_js = alpha_js, P_max = P_max) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }

  #initialize matrix for apply which contains all of the possible grid values
  grid.vals.h = seq(min.h, max.h, length.out = n_grid)
  #apply the previously defined functions to get the CV accuracies at each h value
  if(!is.na(static_train)[1]){
    h.accs = sapply(grid.vals.h,
                    function(x) get.cv.h.static(x))
  }else{
    h.accs = sapply(grid.vals.h,
                    function(x) get.cv.h(x))
  }

  # assign h value based on the one with the largest CV accuracy
  h = h.accs[which.max(h.accs)]
  #h = grid.vals.h[max(which(h.accs==max(h.accs)))]

  guess = nb_updated(scores = scores, classes = classes,
                     prior_g = c(table(classes)/length(classes)),
                     scores_test =  scores_test,
                     s_mat_hat_train = s_mat_hat_train,
                     s_mat_hat_test = s_mat_hat_test, h=h, alpha_js = alpha_js, P_max = P_max,
                     static_train = static_train, static_test = static_test)



  if(return_h){
    return(h)
  }

  #return guesses
  return(guess)

}



#' Generalized additive model function gam() fits the observed binary-valued functional data by fitting a GLMM using a set of known basis functions
#' @param z index z = 1,...,N
#' @param Curves N x D matrix of observed binary series
#' @param tt grid of timepoints going from 0 to 1 with D observations
#' @param k number of basis functions
#' @param bs0 type of basis functions used in the gam
#' @param method method used to evaluate the gam
#' @return Fitted values from the game function for subject z
regression_g = function(z, Curves, tt, k=10, method="REML", bs0 ="cr"){
  z1 = Curves[z,]
  gam1 <- suppressWarnings(gam(z1~s(tt, bs = bs0, m=2, k = k),
              family="binomial", method = method))
  return(gam1$fitted.values)
}


#' Function to estimate eigenfunctions when there are two covariance functions
#' @param K_b : Covariance matrix for the first level component
#' @param K_w : Covariance matrix for the second level component
#' @param pve1 : Proportion of variance explained for the first component
#' @param pve2 : Proportion of variance explained for the second component
#' @param fix_num_of_functions1 if you do not want to used PVE you can fix the number of eigenfunctions to output
#' @param fix_num_of_functions2 if you do not want to used PVE you can fix the number of eigenfunctions to output
#' @return Eigenfunctions and eigenvalues for both types of covariance functions
estimate_eigenfunctions2 = function(K_b, K_w, pve1=0.95, pve2=0.95,
                                    fix_num_of_functions1=NA, fix_num_of_functions2 = NA){

  #SVD
  svd_kb = svd(K_b)
  svd_kw = svd(K_w)

  pw = get_length_pve(pve2, svd_kw$d)
  pb = get_length_pve(pve1, svd_kb$d)

  #Only get the pve length if the number of functions is not given
  if(!is.na(fix_num_of_functions2)){
    if(pw>fix_num_of_functions2){
      pw=fix_num_of_functions2
    }
  }
  if(!is.na(fix_num_of_functions1)){
    if(pb>fix_num_of_functions1){
      pb=fix_num_of_functions1
    }
  }

  eigen_vals1 = svd_kb$d[1:pb]
  eigen_vals2 = svd_kw$d[1:pw]

  eigen_funcs1 = svd_kb$v[,1:pb]
  eigen_funcs2 = svd_kw$v[,1:pw]

  if(pb==1){
    eigen_funcs1 = matrix(eigen_funcs1, ncol = 1)
  }
  if(pw==1){
    eigen_funcs2 = matrix(eigen_funcs2, ncol = 1)
  }

  return(list(eigen_vals1=eigen_vals1, eigen_vals2=eigen_vals2,
              eigen_funcs1=eigen_funcs1, eigen_funcs2=eigen_funcs2))

}



#' Function used in covariance estimation used for
#' @param i cur index of i
#' @param j1 cur index of j
#' @param X_dat Binary Valued Functional Data
#' @param N number of individuals
get_mat_i = function(i,j1, X_dat, N){

  D = dim(X_dat)[2]
  J = dim(X_dat)[1]/N

  j_cur_star = (i-1)*J+j1
  j1_star = (i-1)*J+1
  j2_star = (i-1)*J+J

  d1 = X_dat[j_cur_star, ]
  d2 = X_dat[j1_star:j2_star, ]
  d2 = d2[-j1, ]
  d1 = t(matrix(rep(d1, J-1), ncol=(J-1)))
  d3 = t(d1)%*%d2/length(j1_star:j2_star)
  #K_b=K_b+d3
  return(d3)
  #counter = counter + length(j1_star:j2_star)
}

#' Function used in the estimation of the covariance functions
#' #' Function used in covariance estimation used for
#' @param i.cur cur index
#' @param X_dat Binary Valued Functional Data
#' @param N number of individuals
func.cur = function(i.cur, X_dat, N){

  D = dim(X_dat)[2]
  J = dim(X_dat)[1]/N

  mats = matrix(rowMeans(sapply(1:J, function(x) get_mat_i(i.cur, x, X_dat, N))), ncol = D)
  return(mats)

}






#' Estimate mean function, eigenfunctions, and eigenvalues  for the multilevel model using the Exponential approximation
#' @param X_dat NJ x D matrix of observations
#' @param J Number of realizations per individual
#' @param pve1 Proportion of variance explained for the first component
#' @param pve2 Proportion of variance explained for the second component
#' @param fix_number_of_functions if you do not want to used PVE you can fix the number of eigenfunctions to output
#' @param k number of basis functions in the
#' @param bs0 The basis functions used when smoothing the covariance functions
#' @return Mean function, eigenfunctions and eigenvalues for both types of covariance functions
multilevel_exponential_fpca <-  function(X_dat, J, pve1=0.99, pve2 = 0.99,
                                         fix_number_of_functions = NA,  k = 5, bs0 = "cr"){

  #X_dat = Curves_train
  D = dim(X_dat)[2]
  #Get Individuals
  N = dim(X_dat)[1]/J
  #estimate alpha
  alpha = mean(rowSums(X_dat)!=0)

  #Estimates posting per day
  x_dat_total = t(matrix(rowSums(X_dat), nrow = J))
  #get p_hat_tilde
  alpha_js = apply(x_dat_total!=0 , 2, mean)

  tt = seq(0,1, length.out = D)


  #estimate smooth mean function
  z1 = as.vector(colMeans(X_dat))
  gam1 <- gam(z1~s(tt, bs = "cr", m=2, k = 10),
              family="gaussian", method = "REML")
  #probability needs to ensure values are between 0 and 1
  #also approximation diverges at extremes
  p_hat_t = ifelse(gam1$fitted.values<0.01, 0.01, gam1$fitted.values)
  p_hat_t = ifelse(p_hat_t>0.99, 0.99, p_hat_t)

  p_hat_t = p_hat_t/alpha
  mu_hat = logit(p_hat_t)


  #estimate first level covariance matrix
  K_b = matrix(rowMeans(sapply(1:N, function(x) func.cur(x, X_dat, N))), ncol = D)

  #bivariately smooth the resulting matrix
  #Format for smooth function
  Bs = cbind(as.vector(K_b), rep(1:D, D), as.vector(t(kronecker(matrix(1, ncol=D) , 1:D) )))
  #remove diagonal values
  Bs[which(Bs[,2]==Bs[,3]), 1] = NA
  Bs[,2:3] = Bs[,2:3]/D
  Bs = data.frame(val = Bs[,1], t1 = Bs[,2], t2 = Bs[,3])
  #smooth the covariance matrix, need large theta because we have lots of data points
  beta_ts2 = gam(val~te(t1, t2,
                        bs = "cr", m=2, k=10),
                 method = "REML", family="gaussian", data = Bs)
  newdat = data.frame(t1 = (1:D)/D, t2 = (1:D)/D)
  diag.vals = predict(beta_ts2, newdata = newdat)
  return.mat = matrix(NA, nrow = D, ncol = D)
  diag(return.mat) = diag.vals
  return.mat[which(Bs[,2]!=Bs[,3])] = beta_ts2$fitted.value
  return.mat = ifelse(return.mat<0.0001, 0.0001, return.mat)
  K_b = return.mat

  #Sigma_v = ((1/alpha^2)*K_b-t(t(invlogit(mu_hat)))%*%invlogit(mu_hat))/(t(t(d_logit((mu_hat))))%*%d_logit((mu_hat)))
  Sigma_V = log(K_b/(matrix(p_hat_t, ncol = 1)%*%t(p_hat_t)))
  #Sigma_V = log(K_b/(matrix(p_hat_t, ncol = 1)%*%t(p_hat_t)))*(1/alpha^2)
  #matplot(Sigma_v, type="l")

  #smooth the estimated S_V matrix
  Bs = cbind(as.vector(Sigma_V), rep(1:D, D), as.vector(t(kronecker(matrix(1, ncol=D) , 1:D) )))
  #remove diagonal values
  Bs[which(Bs[,2]==Bs[,3]), 1] = NA
  Bs[,2:3] = Bs[,2:3]/D
  Bs = data.frame(val = Bs[,1], t1 = Bs[,2], t2 = Bs[,3])
  #smooth the covariance matrix, need large theta because we have lots of data points
  beta_ts2 = gam(val~te(t1, t2,
                        bs = "cr", m=2, k=10),
                 method = "REML", family="gaussian", data = Bs)
  newdat = data.frame(t1 = (1:D)/D, t2 = (1:D)/D)
  diag.vals = predict(beta_ts2, newdata = newdat)
  return.mat = matrix(NA, nrow = D, ncol = D)
  diag(return.mat) = diag.vals
  return.mat[which(Bs[,2]!=Bs[,3])] = beta_ts2$fitted.value
  #return.mat = ifelse(return.mat<0.0001, 0.0001, return.mat)
  Sigma_V = return.mat
  #matplot(Sigma_V, type="l")
  #matplot(svd(Sigma_V)$v[,1:3], type="l")

  #total prob mat
  K_w = t((X_dat))%*%X_dat/(dim(X_dat)[1])

  #bivariately smooth the resulting matrix
  #Format for smooth function
  Bs = cbind(as.vector(K_w), rep(1:D, D), as.vector(t(kronecker(matrix(1, ncol=D) , 1:D) )))
  #remove diagonal values
  Bs[which(Bs[,2]==Bs[,3]), 1] = NA
  Bs[,2:3] = Bs[,2:3]/D
  Bs = data.frame(val = Bs[,1], t1 = Bs[,2], t2 = Bs[,3])
  #smooth the covariance matrix, need large theta because we have lots of data points
  beta_ts2 = gam(val~te(t1, t2,
                        bs = "cr", m=2, k=10),
                 method = "REML", family="gaussian", data = Bs)
  newdat = data.frame(t1 = (1:D)/D, t2 = (1:D)/D)
  diag.vals = predict(beta_ts2, newdata = newdat)
  return.mat = matrix(NA, nrow = D, ncol = D)
  diag(return.mat) = diag.vals
  return.mat[which(Bs[,2]!=Bs[,3])] = beta_ts2$fitted.value
  K_w = return.mat
  K_w = make_pos_semi_def(K_w)
  K_w = ifelse(K_w<0.0001, 0.0001, return.mat)

  K_w = log(K_w/(matrix(p_hat_t, ncol = 1)%*%t(p_hat_t)))-Sigma_V

  #Smooth the resulting the estimates
  Bs = cbind(as.vector(K_w), rep(1:D, D), as.vector(t(kronecker(matrix(1, ncol=D) , 1:D) )))
  #remove diagonal values
  Bs[which(Bs[,2]==Bs[,3]), 1] = NA
  Bs[,2:3] = Bs[,2:3]/D
  Bs = data.frame(val = Bs[,1], t1 = Bs[,2], t2 = Bs[,3])
  #smooth the covariance matrix, need large theta because we have lots of data points
  beta_ts2 = gam(val~te(t1, t2,
                        bs = "cr", m=2, k=10),
                 method = "REML", family="gaussian", data = Bs)
  newdat = data.frame(t1 = (1:D)/D, t2 = (1:D)/D)
  diag.vals = predict(beta_ts2, newdata = newdat)
  return.mat = matrix(NA, nrow = D, ncol = D)
  diag(return.mat) = diag.vals
  return.mat[which(Bs[,2]!=Bs[,3])] = beta_ts2$fitted.value
  K_w = return.mat
  K_w = make_pos_semi_def(K_w)
  Sigma_W = round(K_w, 15)
  #matplot(sigma_W, type="l")

  Sigma_z = Sigma_V+Sigma_W

  mu_t = log(p_hat_t)-(diag(Sigma_z))/2-
    log(1+p_hat_t*exp(diag(Sigma_z)))

  fpca_results = estimate_eigenfunctions2(Sigma_V, Sigma_W,  pve1, pve2)

  fpca_results$eigen_funcs1 = fpca_results$eigen_funcs1*sqrt(D)
  fpca_results$eigen_vals1 = fpca_results$eigen_vals1/D

  fpca_results$eigen_funcs2 = fpca_results$eigen_funcs2*sqrt(D)
  fpca_results$eigen_vals2 = fpca_results$eigen_vals2/D

  return(list(eigen_vals1=fpca_results$eigen_vals1,
              eigen_funcs1=fpca_results$eigen_funcs1,
              eigen_vals2=fpca_results$eigen_vals2,
              eigen_funcs2=fpca_results$eigen_funcs2,
              mu_hat = mu_t, alpha = alpha))

}





#' Estimate smooth covariance function
#' @param K_b NJ x D matrix of observations
#' @param k number of basis functions in the
#' @param bs0 The basis functions used when smoothing the covariance functions
#' @return Smooth covariance functions
bivariate_smooth = function(K_b, k = 10, bs0 = "cr"){

  D = dim(K_b)[1]
  #bivariately smooth the resulting matrix
  #Format for smooth function
  Bs = cbind(as.vector(K_b), rep(1:D, D), as.vector(t(kronecker(matrix(1, ncol=D) , 1:D) )))
  #remove diagonal values
  Bs[which(Bs[,2]==Bs[,3]), 1] = NA
  Bs[,2:3] = Bs[,2:3]/D
  Bs = data.frame(val = Bs[,1], t1 = Bs[,2], t2 = Bs[,3])
  #smooth the covariance matrix, need large theta because we have lots of data points
  beta_ts2 = gam(val~te(t1, t2,
                        bs = bs0, m=2, k=10),
                 method = "REML", family="gaussian", data = Bs)
  newdat = data.frame(t1 = (1:D)/D, t2 = (1:D)/D)
  diag.vals = predict(beta_ts2, newdata = newdat)
  return.mat = matrix(NA, nrow = D, ncol = D)
  diag(return.mat) = diag.vals
  return.mat[which(Bs[,2]!=Bs[,3])] = beta_ts2$fitted.value
  K_b = return.mat
  return(K_b)

}

#' Function used to estimate the covariance functions in the multilevel scenario
#' @param i cur index
#' @param j1 cur index
#' @param X_dat.cur Current indexed binary-valued functional data
#' @return estimate averaged over j
K_v_func_over_j = function(i,j1, X_dat.cur){

  #Get the the number of timepoints
  D = dim(X_dat.cur)[2]
  #get the number of days

  #get the row index for the individual and day
  j_cur_star = j1
  #get the row index for the first day of the individual
  j1_star = j1+1
  #get the row index for the last day of the individual
  j2_star = dim(X_dat.cur)[1]

  #get the row of data for the day and individual & scale by the
  d1 = X_dat.cur[j_cur_star, ]
  #get all of the rows for the individual
  d2 = X_dat.cur[j1_star:j2_star, ]
  #to write exception when j1=J-1
  d2 = matrix(d2, ncol = D)
  #get the project of the day j1 and all other days
  d1 = t(matrix(rep(d1, dim(d2)[1]), ncol=dim(d2)[1]))
  d3 = t(d1)%*%(d2)/length(j1_star:j2_star)

  return(d3)

}

#' Function used to estimate the covariance functions in the multilevel scenario
#' @param i.cur cur index
#' @param N number of individuals
#' @param X_dat Current indexed binary-valued functional data
#' @return estimate averaged over j
k_v_func = function(i.cur, X_dat, N){

  D = dim(X_dat)[2]
  J = dim(X_dat)[1]/N

  #get the row index for the first day of the individual
  j1_star = (i.cur-1)*J+1
  #get the row index for the last day of the individual
  j2_star = (i.cur-1)*J+J

  X_dat.cur = X_dat[j1_star:j2_star,]
  X_dat.cur = matrix(X_dat.cur[rowSums(X_dat.cur)>0,], ncol = D)
  J.cur = dim(X_dat.cur)[1]
  #index over the Js
  if(J.cur>1){
    mats = matrix(rowMeans(sapply(1:(J.cur-1), function(x) K_v_func_over_j(i.cur, x, X_dat.cur))), ncol = D)
  }else{
    mats = matrix(0, ncol = D, nrow = D)
  }
  return(mats)

}

#' Function used to estimate the covariance functions in the multilevel scenario
#' @param i cur index
#' @param j1 number of individuals
#' @param X_dat.cur Current indexed binary-valued functional data
#' @return estimate averaged over j
K_w_func_over_j = function(i,j1, X_dat.cur){

  #Get the the number of timepoints
  D = dim(X_dat.cur)[2]
  #get the number of days

  #get the row index for the individual and day
  j_cur_star = j1
  #get the row index for the first day of the individual
  j1_star = j1+1
  #get the row index for the last day of the individual
  j2_star = dim(X_dat.cur)[1]

  #get the row of data for the day and individual & scale by the
  d1 = X_dat.cur[j_cur_star, ]
  #get all of the rows for the individual
  d2 = X_dat.cur[j1_star:j2_star, ]
  #to write exception when j1=J-1
  d2 = matrix(d2, ncol = D)
  #get the project of the day j1 and all other days
  d1 = t(matrix(rep(d1, dim(d2)[1]), ncol=dim(d2)[1]))
  d3 = t(d1-d2)%*%(d1-d2)/length(j1_star:j2_star)

  return(d3)

}

#' Function used to estimate the covariance functions in the multilevel scenario
#' @param i.cur cur index
#' @param N number of individuals
#' @param X_dat Current indexed binary-valued functional data
#' @return estimate averaged over j
k_w_func = function(i.cur, X_dat, N){

  D = dim(X_dat)[2]
  J = dim(X_dat)[1]/N

  #get the row index for the first day of the individual
  j1_star = (i.cur-1)*J+1
  #get the row index for the last day of the individual
  j2_star = (i.cur-1)*J+J

  X_dat.cur = X_dat[j1_star:j2_star,]
  X_dat.cur = matrix(X_dat.cur[rowSums(X_dat.cur)>0,], ncol = D)
  J.cur = dim(X_dat.cur)[1]
  #index over the Js
  if(J.cur>1){
    mats = matrix(rowMeans(sapply(1:(J.cur-1), function(x) K_w_func_over_j(i.cur, x, X_dat.cur))), ncol = D)
  }else{
    mats = matrix(0, ncol = D, nrow = D)
  }
  return(mats/2)

}


#' Estimate mean function, eigenfunctions, and eigenvalues  for the multilevel model using the Linear approximation
#' @param X_dat NJ x D matrix of observations
#' @param J Number of realizations per individual
#' @param pve1 Proportion of variance explained for the first component
#' @param pve2 Proportion of variance explained for the second component
#' @param fix_number_of_functions if you do not want to used PVE you can fix the number of eigenfunctions to output
#' @param k number of basis functions in the
#' @param bs0 The basis functions used when smoothing the covariance functions
#' @return Mean function, eigenfunctions and eigenvalues for both types of covariance functions
multilevel_linear_fpca = function(X_dat, J, pve1=0.95, pve2 = 0.95,
                                  fix_number_of_functions = NA,  k = 5, bs0 = "cr"){

  #X_dat = Curves_train
  D = dim(X_dat)[2]
  #Get Individuals
  N = dim(X_dat)[1]/J
  #estimate alpha
  alpha = mean(rowSums(X_dat)!=0)

  #Estimates posting per day
  x_dat_total = t(matrix(rowSums(X_dat), nrow = J))
  #get p_hat_tilde
  alpha_js = apply(x_dat_total!=0 , 2, mean)

  tt = seq(0,1, length.out = D)

  #get number of days which the user posted
  #Js_s = apply(matrix(s_mat, nrow = N), 1, sum)

  get_EXi = function(j, X_dat, J){

    N = dim(X_dat)[1]/J
    js = (1:N-1)*J+j
    colMeans(X_dat[js,])

  }

  E.X_i.hat = sapply(1:J, function(x) get_EXi(x, X_dat, J))
  norm_vec <- function(x) sqrt(sum(x^2))

  #from equation 13
  #z1 = 1/(norm_vec(alpha_js)^2)*E.X_i.hat %*% matrix(alpha_js, ncol = 1)
  #ensure probability not at extremes 0 and 1

  z1 = colMeans(X_dat[rowSums(X_dat)>0,])

  z1 = ifelse( z1 <0.01, 0.01, z1)
  z1 = ifelse(z1>0.99, 0.99, z1)
  #obtain underlying process and smooth
  z1 = logit(z1)

  #smooth mean function
  gam1 <- gam(z1~s(tt, bs = bs0, m=2, k = 10),
              family="gaussian", method = "REML")
  mu_hat = gam1$fitted.values
  #obtain smoothed probability function
  p_hat_t = invlogit(mu_hat)


  #estimate first level covariance matrix
  K_v = matrix(rowMeans(sapply(1:N, function(x) k_v_func(x, X_dat, N) )), ncol = D)
  # bivariately smooth matrix
  K_v = bivariate_smooth(K_v, k = k, bs0 = bs0)
  Sigma_v = (K_v-t(t(invlogit(mu_hat)))%*%invlogit(mu_hat))/(t(t(d_logit((mu_hat))))%*%d_logit((mu_hat)))
  #matplot(Sigma_v, type="l")
  Sigma_v = make_pos_semi_def(Sigma_v)
  Sigma_v = bivariate_smooth(Sigma_v, k = 5)
  Sigma_v = round(Sigma_v, 15)

  K_w = matrix(rowMeans(sapply(1:N, function(x) k_w_func(x, X_dat, N) )), ncol = D)
  K_w2= bivariate_smooth(K_w, k = 10)
  #matplot(K_w2,type="l")
  Sigma_w = K_w/((t(t(d_logit((mu_hat))))%*%d_logit((mu_hat))))
  #Sigma_w = bivariate_smooth(Sigma_w, k = 10)
  Sigma_w = make_pos_semi_def(Sigma_w)
  Sigma_w = bivariate_smooth(Sigma_w, k = k, bs0 = bs0)
  Sigma_w = round(Sigma_w, 15)

  Sigma_z = Sigma_v+Sigma_w

  fpca_results = estimate_eigenfunctions2(Sigma_v, Sigma_w,  pve1, pve2,
                                          fix_num_of_functions1 = fix_number_of_functions,
                                          fix_num_of_functions2 = fix_number_of_functions)

  fpca_results$eigen_funcs1 = fpca_results$eigen_funcs1*sqrt(D)
  fpca_results$eigen_vals1 = fpca_results$eigen_vals1/D

  fpca_results$eigen_funcs2 = fpca_results$eigen_funcs2*sqrt(D)
  fpca_results$eigen_vals2 = fpca_results$eigen_vals2/D

  return(list(eigen_vals1=fpca_results$eigen_vals1,
              eigen_funcs1=fpca_results$eigen_funcs1,
              eigen_vals2=fpca_results$eigen_vals2,
              eigen_funcs2=fpca_results$eigen_funcs2,
              mu_hat = mu_hat, alpha_js = alpha_js))

}





#' Solve the GLMM based on the Estimates of the scores for the multilevel scenario
#' @param data_formatted NJ x D binary matrix
#' @param J Number of curves per subject
#' @param I N Number of subjects
#' @param eigen_vals1 Vector  of Eigenvalues of the first level (individual) effects
#' @param eigen_vals2 Vector of Eigenvalues of the second level (daily) effects
#' @param eigen_funcs1 Matrix K1 x D of eigenfunctions of first level
#' @param eigen_funcs2 Matrix K2 x D of eigenfunctions of second level
#' @param mu_t Mean function vector
#' @param s_mat Matrix of estimated latent state values S
#' @return First Level Scores for every individual
estimate_scores = function(data_formatted, s_mat, J, I,
                           eigen_vals1, eigen_vals2, eigen_funcs1,
                           eigen_funcs2, mu_t){


  #beta0_hat=colMeans(data_formatted)
  D=dim(data_formatted)[2]

  s_mat2 = matrix(t(s_mat), ncol = 1)
  data_formatted = data_formatted[which(s_mat2==1),]

  N = I
  #get number of days which the user posted
  Js_s = apply(matrix(s_mat, nrow = N), 1, sum)

  #set up the functions and eigenvalues
  psi1_est = eigen_funcs1
  psi2_est = eigen_funcs2
  lambda1_est = eigen_vals1
  lambda2_est = eigen_vals2

  bayesglm_daily = function(i, Js_s){

    J_cur = Js_s[i]
    if(i==1){
      J_start = 0
    }else{
      J_start = sum(Js_s[1:(i-1)])
    }

    J_cur1 = J_start+1
    J_cur2 = J_start+J_cur


    #Prepare matrices for the glm
    dc1 = kronecker(matrix(1, ncol=1, nrow=J_cur), psi1_est)
    for(i in 1:ncol(psi2_est)){
      if(i == 1){
        dc2 = kronecker(diag(J_cur), psi2_est[,i])
      }else{
        dc2 = cbind(dc2, kronecker(diag(J_cur), psi2_est[,i]))
      }
    }

    #Repeat mean functions accross the J days
    alpha_t = mu_t
    alpha_t_hat = as.vector(alpha_t)


    beta0_hat=matrix(rep(alpha_t_hat, J_cur), ncol=1)

    #Set up the variance
    scales_test=c(lambda1_est,
                  rep(lambda2_est, J_cur))

    #set up names for the glm
    c1_names=c()
    c2_names=c()
    for(i in 1:ncol(psi1_est)){
      c1_names=c(c1_names, paste("Z", i, sep=""))
    }
    for(i in 1:ncol(psi2_est)){
      for(j in 1:J_cur){
        c2_names=c(c2_names, paste("W", i, "_", j,  sep=""))
      }
    }

    names_testings=c("Y", "beta0",
                     c1_names, c2_names)

    formula.1="Y ~ -1 + offset(beta0)"
    for(i in 3:length(names_testings)){
      formula.1=paste(formula.1, " + ", names_testings[i],
                      sep="")
    }

    #if(i%%100==0){
    #  print(paste("On Iteration ", i , " out of ", I))
    #}

    cur_user_dat = as.vector(t(data_formatted[J_cur1:J_cur2, ]))
    testing_data=data.frame(cur_user_dat, beta0_hat, dc1, dc2)
    names(testing_data)=names_testings

    m.bayes <- bayesglm(formula.1,
                        data=testing_data, family = binomial,
                        prior.scale = sqrt(scales_test), #Set sd values
                        prior.df = Inf, #Set Normal
                        scaled = F) #Do not scale the scale parameter

    return(m.bayes$coefficients[1:ncol(psi1_est)])
    #return(m.bayes$coefficients)
  }

  #vec = 1:I
  #ind_scores = mclapply(vec, function(x) bayesglm_daily(x),
  #                              mc.cores = 16)

  vec = matrix(1:I, ncol = 1)
  ind_scores  = t(apply(vec, 1, function(x) bayesglm_daily(x, Js_s)))

  #vec_list = list()
  #for(i in 1:I){
  #  vec_list[[i]]=i
  #}
  #ind_scores  = lapply(vec_list, bayesglm_daily)

  return(ind_scores)

}


#' Solve the LMM based on the Estimates of the scores for the multilevel scenario
#' @param data_formatted NJ x D matrix
#' @param J Number of curves per subject
#' @param I N Number of subjects
#' @param eigen_vals1 Vector  of Eigenvalues of the first level (individual) effects
#' @param eigen_vals2 Vector of Eigenvalues of the second level (daily) effects
#' @param eigen_funcs1 Matrix K1 x D of eigenfunctions of first level
#' @param eigen_funcs2 Matrix K2 x D of eigenfunctions of second level
#' @param mu_t mean function defined over D
#' @param s_mat Matrix of estimated latent state values S
#' @return First Level Scores for every individual
estimate_scores_normal = function(data_formatted, s_mat, J, I,
                                  eigen_vals1, eigen_vals2, eigen_funcs1,
                                  eigen_funcs2, mu_t){


  #beta0_hat=colMeans(data_formatted)
  D=dim(data_formatted)[2]

  s_mat2 = matrix(t(s_mat), ncol = 1)
  data_formatted = data_formatted[which(s_mat2==1),]

  N = I
  #get number of days which the user posted
  Js_s = apply(matrix(s_mat, nrow = N), 1, sum)

  #set up the functions and eigenvalues
  psi1_est = eigen_funcs1
  psi2_est = eigen_funcs2
  lambda1_est = eigen_vals1
  lambda2_est = eigen_vals2

  bayesglm_daily = function(i, Js_s){

    J_cur = Js_s[i]
    if(i==1){
      J_start = 0
    }else{
      J_start = sum(Js_s[1:(i-1)])
    }

    J_cur1 = J_start+1
    J_cur2 = J_start+J_cur


    #Prepare matrices for the glm
    dc1 = kronecker(matrix(1, ncol=1, nrow=J_cur), psi1_est)
    for(i in 1:ncol(psi2_est)){
      if(i == 1){
        dc2 = kronecker(diag(J_cur), psi2_est[,i])
      }else{
        dc2 = cbind(dc2, kronecker(diag(J_cur), psi2_est[,i]))
      }
    }

    #Repeat mean functions accross the J days
    alpha_t = mu_t
    alpha_t_hat = as.vector(alpha_t)


    beta0_hat=matrix(rep(alpha_t_hat, J_cur), ncol=1)

    #Set up the variance
    scales_test=c(lambda1_est,
                  rep(lambda2_est, J_cur))

    #set up names for the glm
    c1_names=c()
    c2_names=c()
    for(i in 1:ncol(psi1_est)){
      c1_names=c(c1_names, paste("Z", i, sep=""))
    }
    for(i in 1:ncol(psi2_est)){
      for(j in 1:J_cur){
        c2_names=c(c2_names, paste("W", i, "_", j,  sep=""))
      }
    }

    names_testings=c("Y", "beta0",
                     c1_names, c2_names)

    formula.1="Y ~ -1 + offset(beta0)"
    for(i in 3:length(names_testings)){
      formula.1=paste(formula.1, " + ", names_testings[i],
                      sep="")
    }

    #if(i%%100==0){
    #  print(paste("On Iteration ", i , " out of ", I))
    #}

    cur_user_dat = as.vector(t(data_formatted[J_cur1:J_cur2, ]))
    testing_data=data.frame(cur_user_dat, beta0_hat, dc1, dc2)
    names(testing_data)=names_testings

    m.bayes <- bayesglm(formula.1,
                        data=testing_data, family = gaussian(),
                        prior.scale = sqrt(scales_test), #Set sd values
                        prior.df = Inf, #Set Normal
                        scaled = F) #Do not scale the scale parameter

    return(m.bayes$coefficients[1:ncol(psi1_est)])
    #return(m.bayes$coefficients)
  }

  #vec = 1:I
  #ind_scores = mclapply(vec, function(x) bayesglm_daily(x),
  #                              mc.cores = 16)

  vec = matrix(1:I, ncol = 1)
  ind_scores  = t(apply(vec, 1, function(x) bayesglm_daily(x, Js_s)))

  #vec_list = list()
  #for(i in 1:I){
  #  vec_list[[i]]=i
  #}
  #ind_scores  = lapply(vec_list, bayesglm_daily)

  return(ind_scores)

}





#'Function: Grid search to estimate predicted values and estimate the values of h
#' @param scores N x K matrix of scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param h multiplier for kernel based density function
#' @return predicted classes for the Bayes Classifier
nb_updated_scores_only = function(scores, classes, prior_g, scores_test, h = 1.06){

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  #get K
  nd = dim(scores)[2]
  #get number of groups
  ng = length(unique(classes))
  #initialize list
  densities = list()
  #estimate density at each K for each group
  for(i in 1:ng){
    densities[[i]]=list()
    for(k in 1:nd){
      densities[[i]][[k]] = density(scores[classes==i,k],
                                    kernel = "gaussian",
                                    bw = h*sd(scores[classes==i,k]))
    }
  }

  # get number of test functions
  n_test = dim(scores_test)[1]
  p.mat = matrix(NA, nrow = n_test, ncol = length(prior_g))
  vec = matrix(1:n_test, ncol = 1)
  #For each group get the Bayes classifier value of probability of being in that group
  for(i in 1:ng){
    #apply for each user get the probability for each component in that group
    pdf.vals.test.1 = t(apply(vec, 1,
                              function(x)  get_pdf_den2(densities, scores_test, rep(i, n_test), x)))
    #apply for each user to get the product of those K probabilities
    pdf_vals = apply(pdf.vals.test.1, 1, prod)
    #multiply by prior probability
    p.mat[,i] = prior_g[i]* pdf_vals #p.mat is now matrix of Bayes probabilities
  }

  #get guess based on which is the max
  guess = apply(p.mat, 1, which.max)

  return(guess)

}



#'Function: Grid search to estimate predicted values and estimate the values of h
#' @param scores N x K matrix of 1st level scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param min.h min possible value for the multiplier
#' @param max.h maximum possible value for the multiplier
#' @param n_grid number of values between min.h and max.h to search over
#' @param CV Number of folds for cross validation
#' @param return_h T/F to return the value of the multiplier
#' @param return_prob T/F to return group Bayes classifier probability for each individual in testing set
#' @return predictions from the grid search
nb_updated_grid_scores_only = function(scores, classes, prior_g, scores_test,
                                       min.h = 0.01, max.h = 1.5,
                                       CV = 10, n_grid = 10, return_h = F, return_prob = F){

  #create matrix for apply functions
  vec.cv = matrix(1:CV, ncol = 1)

  #create vector of possible CV groups to each account
  #add extra incase unequal distribution of groups
  cvgroups = rep(1:CV, (length(classes)/CV+1))
  #remove the unneeded values
  cvgroups = cvgroups[1:length(classes)]
  #randomly assign CV group to each account
  cvgroups = sample(cvgroups, size = length(classes), replace = F)

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    #If k == 1 change to vector
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  # define function here to use the cvgroups this function
  # will be used in the following apply statement
  get.cv.h = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated_scores_only(scores[cvgroups!=x,], classes[cvgroups!=x],
                                 c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                                 scores[cvgroups==x,], h = h.val) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }

  #initialize matrix for apply which contains all of the possible grid values
  grid.vals.h = matrix(seq(min.h, max.h, length.out = n_grid), ncol = 1)
  #apply the previously defined functions to get the CV accuracies at each h value
  h.accs = apply(grid.vals.h, 1, function(h) get.cv.h(h))

  # assign h value based on the one with the largest CV accuracy
  h = grid.vals.h[which.max(h.accs)]
  #h = grid.vals.h[max(which(h.accs==max(h.accs)))]

  #get K
  nd = dim(scores)[2]
  #get number of groups
  ng = length(unique(classes))
  #initialize list
  densities = list()
  #estimate density at each K for each group
  for(i in 1:ng){
    densities[[i]]=list()
    for(k in 1:nd){
      densities[[i]][[k]] = density(scores[classes==i,k],
                                    kernel = "gaussian",
                                    bw = h*sd(scores[classes==i,k]))
    }
  }

  # get number of test functions
  n_test = dim(scores_test)[1]
  p.mat = matrix(NA, nrow = n_test, ncol = length(prior_g))
  vec = matrix(1:n_test, ncol = 1)
  #For each group get the Bayes classifier value of probability of being in that group
  for(i in 1:ng){
    #apply for each user get the probability for each component in that group
    pdf.vals.test.1 = t(apply(vec, 1,
                              function(x)  get_pdf_den2(densities, scores_test, rep(i, n_test), x)))
    #apply for each user to get the product of those K probabilities
    pdf_vals = apply(pdf.vals.test.1, 1, prod)
    #multiply by prior probability
    p.mat[,i] = prior_g[i]* pdf_vals #p.mat is now matrix of Bayes probabilities
  }

  #returns matrix of probabilies for each group
  if(return_prob){
    return(p.mat/rowSums(p.mat))
  }

  #group prediction is based on maximum posterior probability
  guess = apply(p.mat, 1, which.max)

  #print(h)

  if(return_h){
    return(h)
  }

  #return guesses
  return(guess)

}


#' Function: wrapper function for bayesglm() which outputs expected coefficients for the gaussian responses
#' @param z index the n individual
#' @param dta : data frame contain the subject id, mean function, eigenfunctions and observe binary values
#' @param lm_structure formula displaying the bayesglm function
#' @param eigen_vals1 eigen_values for the psi coefficients
#' @return Fitted values from the game function for subject z
regression_bf2 = function(z, dta, lm_structure, eigen_vals1){
  bayesglm1 = bayesglm(lm_structure,
                       family = binomial(link = "logit"),
                       data = subset(dta, dta$id==z),
                       prior.scale = sqrt(eigen_vals1), #set scales
                       prior.df = Inf, #normal priors
                       scaled = F ) #Do not adjust the scales
  return(bayesglm1$coefficients)
}

#' function to get basis for simulation study
#' @param scenario Scenario used to simulate the data
#' @param D number of points in the grid
get_true_bases = function(scenario=1, D){

  if(scenario == 1){

    grid = seq(0, 1, length = D)

    #true mean
    mu_t = -0.5-sin(2*pi*grid)-cos(2*pi*grid)

    #set variance struture or scneario A
    theta1<- theta0 <- 1/(c(1:40)^2)
    theta1[c(1, 8:40)] = 0
    theta0[c(1, 8:40)] = 0

    theta = theta1
    K = 40
    K <- length(theta); Khalf <- round(K/2)
    Kseqeven <- 2*(1:Khalf); Kseqodd<- Kseqeven-1
    Phitt1 <- sapply(c(1:Khalf), function(j) sqrt(2) * sin(2*pi*j*grid))
    Phitt2 <- sapply(c(1:Khalf), function(j) sqrt(2) * cos(2*pi*j*grid))
    Phitt <- matrix(0, ncol=2*Khalf, nrow=length(grid))
    Phitt[,Kseqeven] <- Phitt1
    Phitt[,Kseqodd] <- Phitt2
    Phitt <-cbind(rep(1, length(grid)), Phitt)[, 1:K]

    mu0 <-  c(0, -0.5, 1, -0.5, 1, -0.5)
    mu1 <- c(0, -0.75, 0.75, -0.15, 1.4, 0.1)
    mu = (mu1+mu0)/2
    mu_t = mu_t+Phitt[,1:6]%*%mu

    eigen_vals1 = theta1[which(theta1!=0)]
    eigen_funcs1 = Phitt[,which(theta1!=0)]

    #Second Level eigenfunctions
    psi_21 = sqrt(2)*cos(4*pi*grid)
    psi_22 = sqrt(2)*sin(4*pi*grid)
    psi_2=cbind(psi_21, psi_22)

    lambda2=matrix(c(0.25, 0.08), nrow=1)

    eigen_vals2 = lambda2
    eigen_funcs2 = psi_2

  }
  if(scenario == 2){

    grid = seq(0, 1, length = D)

    #true mean
    mu_t = -0.5-sin(2*pi*grid)-cos(2*pi*grid)

    #set variance struture or scneario A
    theta1<- theta0 <- 1/(c(1:40)^2)
    theta1[c(1, 8:40)] = 0
    theta0[c(1, 8:40)] = 0

    theta = theta1
    K = 40
    K <- length(theta); Khalf <- round(K/2)
    Kseqeven <- 2*(1:Khalf); Kseqodd<- Kseqeven-1
    Phitt1 <- sapply(c(1:Khalf), function(j) sqrt(2) * sin(2*pi*j*grid))
    Phitt2 <- sapply(c(1:Khalf), function(j) sqrt(2) * cos(2*pi*j*grid))
    Phitt <- matrix(0, ncol=2*Khalf, nrow=length(grid))
    Phitt[,Kseqeven] <- Phitt1
    Phitt[,Kseqodd] <- Phitt2
    Phitt <-cbind(rep(1, length(grid)), Phitt)[, 1:K]

    mu0 <-  c(0, -0.5, 1, -0.5, 1, -0.5)
    mu1 <- c(0, -0.75, 0.75, -0.15, 1.4, 0.1)

    mu0 <-  -1*theta1[1:8]/2
    mu1 <- theta0[1:8]/2

    mu = (mu1+mu0)/2
    mu_t = mu_t+Phitt[,1:length(mu)]%*%mu

    eigen_vals1 = theta1[which(theta1!=0)]
    eigen_funcs1 = Phitt[,which(theta1!=0)]

    #Second Level eigenfunctions
    psi_21 = rep(1/sqrt(D), D)
    psi_2= matrix(psi_21, ncol = 1)

    lambda2 = matrix(c(0.25), nrow=1)

    eigen_vals2 = lambda2
    eigen_funcs2 = psi_2

  }
  if(scenario == 3){

    grid = seq(0, 1, length = D)

    #true mean
    mu_t = -0.5-sin(2*pi*grid)-cos(2*pi*grid)

    #set variance struture or scneario A
    theta0 <- exp(-c(1:40)/3)
    theta1 <- exp(-c(1:40)/2)
    theta1[c(1, 8:40)] = 0
    theta0[c(1, 8:40)] = 0

    theta = theta1
    K = 40
    K <- length(theta); Khalf <- round(K/2)
    Kseqeven <- 2*(1:Khalf); Kseqodd<- Kseqeven-1
    Phitt1 <- sapply(c(1:Khalf), function(j) sqrt(2) * sin(2*pi*j*grid))
    Phitt2 <- sapply(c(1:Khalf), function(j) sqrt(2) * cos(2*pi*j*grid))
    Phitt <- matrix(0, ncol=2*Khalf, nrow=length(grid))
    Phitt[,Kseqeven] <- Phitt1
    Phitt[,Kseqodd] <- Phitt2
    Phitt <-cbind(rep(1, length(grid)), Phitt)[, 1:K]

    mu0 <-  rep(0, 8)
    mu1 <- rep(0, 8)
    mu = (mu1+mu0)/2
    mu_t = mu_t+Phitt[,1:length(mu)]%*%mu

    eigen_vals1 = theta1[which(theta1!=0)]
    eigen_funcs1 = Phitt[,which(theta1!=0)]

    #Second Level eigenfunctions
    psi_21 = sqrt(2)*cos(4*pi*grid)
    psi_22 = sqrt(2)*sin(4*pi*grid)
    psi_2=cbind(psi_21, psi_22)

    lambda2=matrix(c(0.25, 0.08), nrow=1)

    eigen_vals2 = lambda2
    eigen_funcs2 = psi_2

  }

  return(list(eigen_vals1=eigen_vals1,
              eigen_funcs1=eigen_funcs1,
              eigen_vals2=eigen_vals2,
              eigen_funcs2=eigen_funcs2,
              mu_hat = mu_t))



}

#' function to get true properties of S for the simulation study
#' @param scenario Scenario used to simulation the data
#' @param J number of realizations of functional data
get_true_alpha = function(scenario = 1, J){


  #constant
  if(scenario == 1){

    alpha1 = 0.4
    alpha2 = 0.6
    alpha_js = rbind(rep(alpha2, J), rep(alpha1, J))
    return(alpha_js)
  }
  if(scenario == 2){

    alpha1 = 0.5
    alpha2 = 0.25

    alpha_js = rbind(c(rep(c(alpha1, alpha2), ceiling(J/2))[1:J]),
                     c(rep(alpha1, floor(J/2)), rep(alpha2, J-floor(J/2))))
    return(alpha_js)

  }
  if(scenario == 3){

    alpha1 = 2/3
    Js = 1:J
    alpha_js = rbind(2/3-(1/Js)*(1/4), rep(alpha1, J))
    return(alpha_js)


  }
  if(scenario == 4){

    alpha1 = c(0.4543210 , 0.4098765 , 0.4370370 , 0.4197531 , 0.3975309 , 0.4345679 , 0.4246914 , 0.4222222,
               0.3851852 , 0.3925926 , 0.4543210 , 0.4493827 , 0.4543210 , 0.4469136 , 0.4098765 , 0.3950617,
               0.4024691 , 0.4493827 , 0.4617284 , 0.4814815 , 0.4296296 , 0.4518519 , 0.4395062 , 0.4617284,
               0.4395062 , 0.4666667 , 0.4567901 , 0.4716049 , 0.4320988 , 0.4592593 , 0.4592593 , 0.4790123,
               0.4938272 , 0.5037037 , 0.5037037 , 0.4888889 , 0.4691358 , 0.4617284 , 0.4740741 , 0.5358025,
               0.5358025 , 0.5209877 , 0.5135802 , 0.5012346 , 0.5234568 , 0.5185185 , 0.5111111 , 0.5555556,
               0.5407407 , 0.5407407)
    alpha2 = c(0.26824458 , 0.36982249 , 0.08974359 , 0.36883629 , 0.11143984 , 0.40729783 , 0.09861933 , 0.56213018,
               0.37080868 , 0.17751479 , 0.06213018 , 0.13214990 , 0.08678501 , 0.09566075 , 0.19526627 , 0.16370809,
               0.26331361 , 0.35305720 , 0.23372781 , 0.26923077 , 0.37771203 , 0.39743590 , 0.18639053 , 0.20216963,
               0.14003945 , 0.29881657 , 0.16863905 , 0.18441815 , 0.25345168 , 0.32741617 , 0.21794872 , 0.17357002,
               0.06607495 , 0.14990138 , 0.16962525 , 0.27514793 , 0.10946746 , 0.24260355 , 0.20019724 , 0.18244576,
               0.26725838 , 0.23076923 , 0.24358974 , 0.19428008 , 0.22879684 , 0.15976331 , 0.32248521 , 0.18836292,
               0.24161736 , 0.52761341)

    alpha1 = alpha1[1:J]
    alpha2 = alpha2[1:J]

    alpha_js = rbind(alpha2, alpha1)
    return(alpha_js)

  }
  if(scenario == 5){

    alpha1 = 0.5
    alpha2 = 0.5
    alpha_js = rbind(rep(alpha2, J), rep(alpha1, J))
    return(alpha_js)
  }
  if(scenario == 6){

    alpha1 = c(0.4543210 , 0.4098765 , 0.4370370 , 0.4197531 , 0.3975309 , 0.4345679 , 0.4246914 , 0.4222222,
               0.3851852 , 0.3925926 , 0.4543210 , 0.4493827 , 0.4543210 , 0.4469136 , 0.4098765 , 0.3950617,
               0.4024691 , 0.4493827 , 0.4617284 , 0.4814815 , 0.4296296 , 0.4518519 , 0.4395062 , 0.4617284,
               0.4395062 , 0.4666667 , 0.4567901 , 0.4716049 , 0.4320988 , 0.4592593 , 0.4592593 , 0.4790123,
               0.4938272 , 0.5037037 , 0.5037037 , 0.4888889 , 0.4691358 , 0.4617284 , 0.4740741 , 0.5358025,
               0.5358025 , 0.5209877 , 0.5135802 , 0.5012346 , 0.5234568 , 0.5185185 , 0.5111111 , 0.5555556,
               0.5407407 , 0.5407407)
    alpha2 = c(0.26824458 , 0.36982249 , 0.08974359 , 0.36883629 , 0.11143984 , 0.40729783 , 0.09861933 , 0.56213018,
               0.37080868 , 0.17751479 , 0.06213018 , 0.13214990 , 0.08678501 , 0.09566075 , 0.19526627 , 0.16370809,
               0.26331361 , 0.35305720 , 0.23372781 , 0.26923077 , 0.37771203 , 0.39743590 , 0.18639053 , 0.20216963,
               0.14003945 , 0.29881657 , 0.16863905 , 0.18441815 , 0.25345168 , 0.32741617 , 0.21794872 , 0.17357002,
               0.06607495 , 0.14990138 , 0.16962525 , 0.27514793 , 0.10946746 , 0.24260355 , 0.20019724 , 0.18244576,
               0.26725838 , 0.23076923 , 0.24358974 , 0.19428008 , 0.22879684 , 0.15976331 , 0.32248521 , 0.18836292,
               0.24161736 , 0.52761341)

    alpha1 = alpha1[1:J]
    alpha2 = alpha2[1:J]

    alpha_js = rbind(alpha2, alpha1)
    return(alpha_js)

  }



}


#' function used to train the gAR models
#' @param l the group which is currently being used to fit gAR models
#' @param j_lags the lag used in the gAR
#' @param s_mat_train estimated latent state data used to fit the gAR model
#' @param aic T/F to return the AIC of the model
#' @param classes vector of the classes
#' @param static_train dataframe containing the information of covariates which one would like to include in the gAR model
#' @return Fitted gAR model
fit_ajs_model = function(l, j_lags, s_mat_train, aic = F, classes, static_train=NA){

  N = dim(s_mat_train)[1]
  J = dim(s_mat_train)[2]

  data.s2 = cbind.data.frame(S_dat = c(t(s_mat_train)),
                             ts = rep(1:J, N),
                             class = rep(classes, each = J),
                             individual = as.factor(rep(1:N, each = J)) )
  if(!is.na(static_train)[1]){
    covariates = static_train[rep(seq_len(nrow(static_train)), each = J),]
    data.s2 = cbind.data.frame(data.s2,  covariates)
  }

  if(j_lags == 0){

    data.s.cur2 = subset(data.s2, data.s2$class==l)
    data.s.cur2 = subset(data.s.cur2, complete.cases(data.s.cur2))

    if(!is.na(static_train)[1]){

      cur.names = names(data.s2)
      cov.names = cur.names[(length(cur.names)-dim(static_train)[2]+1):(length(cur.names))]
      cov.names.formatted = paste(cov.names, sep= "", collapse = " + ")

      formula.cur = paste("S_dat ~ 1 + ",  cov.names.formatted , sep= "")


      m.bayes <- bayesglm(formula.cur, family = binomial, data = data.s.cur2,
                          prior.df = Inf, scaled = FALSE)

    }else{

      m.bayes <- bayesglm(S_dat ~ 1, family = binomial, data = data.s.cur2,
                          prior.df = Inf, scaled = FALSE)

    }
    if(aic){
      #return(m.bayes$aic/(dim(data.s.cur2)[1]))
      return(m.bayes$aic)
    }
    return(m.bayes)

  }





  for(j in 1:j_lags){
    lagj = c(rep(NA, j), c(t(s_mat_train))[1:(N*J-j)])
    data.s2 = cbind.data.frame(data.s2 , lagj)
  }

  data.s.cur2 = subset(data.s2, data.s2$class==l)
  data.s.cur2 = subset(data.s.cur2, complete.cases(data.s.cur2))
  data.s.cur2 = subset(data.s.cur2, data.s.cur2$ts > j_lags)


  c1_names=c()
  for(i in 1:j_lags){
    c1_names=c(c1_names, paste("S", i, sep=""))
  }

  names(data.s.cur2) = c(names(data.s.cur2)[1:(length(names(data.s.cur2))-j_lags)],
                         c1_names)

  formula.cur = paste("S_dat ~ ")
  for(j in 1:j_lags){
    if(j==1){
      formula.cur = paste(formula.cur, c1_names[j], sep= "")
    }else{
      formula.cur = paste(formula.cur,  "+", c1_names[j], sep= "")
    }
  }

  if(!is.na(static_train)[1]){

    cov.names = names(covariates)
    #cov.names = cur.names[(length(cur.names)-dim(static_train)[2]+1):(length(cur.names))]
    cov.names.formatted = paste(cov.names, sep= "", collapse = " + ")

    formula.cur = paste(formula.cur, " + ",  cov.names.formatted , sep= "")

  }

  m.bayes <- bayesglm(formula.cur, family = binomial, data = data.s.cur2,
                      prior.df = Inf, scaled = FALSE)


  a.int = 2
  initial_probs = rep(NA, 2^j_lags)
  initial_combos = as.matrix.data.frame(expand.grid(rep(list(0:1),j_lags)))
  if(j_lags ==1 ){
    mat.obs = matrix(data.s.cur2[ , (dim(data.s.cur2)[2]-j_lags+1):dim(data.s.cur2)[2]], ncol = j_lags)
    mat.obs = matrix(mat.obs[complete.cases(mat.obs),], ncol = j_lags)
  }else{
    #take the columns of interest & whether to look at first p_l day or all days for initial probabilities
    #mat.obs = data.s.cur2[which(data.s.cur2$ts==1) , (dim(data.s.cur2)[2]-j_lags+1):dim(data.s.cur2)[2]]
    mat.obs = data.s.cur2[ , (dim(data.s.cur2)[2]-j_lags+1):dim(data.s.cur2)[2]]
    mat.obs = mat.obs[complete.cases(mat.obs),]
  }

  initial_probs = apply(initial_combos, 1, function(x) sum(apply(mat.obs, 1, function(y) sum(y==x)==j_lags ) ) )
  initial_probs = (initial_probs+a.int)/(sum(initial_probs)+a.int*(2^j_lags))
  #change for tuniting paramter
  #initial_probs = (initial_probs)/(sum(initial_probs))


  if(aic){
    #return(m.bayes$aic/(dim(data.s.cur2)[1]))
    #return(m.bayes$aic)
    return(BIC(m.bayes))
  }
  return(list(model = m.bayes, initial_probs=initial_probs, initial_combos = initial_combos))
}


#' function used to predict the group for new individuals based on the latent states and the fitted gAR models
#' @param i index of the current individual
#' @param l the group which is currently being used
#' @param p the lag of the current models
#' @param models_ls Fitted gAR models for the various groups
#' @param s_mat_test estimated latent state data used to predict the gAR model
#' @param static_test dataframe containing the information of covariates which one would like to include in the gAR model
#' @return Predicted probability of being in group l for the new individuals
predict_new_ind_group_model = function(i, l, p, models_ls, s_mat_test, static_test = NA){

  N_test = dim(s_mat_test)[1]
  J = dim(s_mat_test)[2]

  #p = get_ps[l]
  model_cur = models_ls[[l]]$model
  initial_probs = models_ls[[l]]$initial_probs
  s_mat_cur = s_mat_test[i, ]
  #data.s.cur = cbind.data.frame(S_dat = s_mat_cur)
  data.s.cur = cbind.data.frame(S_dat = s_mat_cur,
                                dow = as.factor(rep(1:7, ceiling(J/7))[1:J]))

  if(!is.na(static_test)[1]){
    # data.s.cur = cbind.data.frame(data.s.cur,  followers = rep(static_test$followers_count[i], each = J),
    #                               friends = rep(static_test$friends_count[i], each = J))
    covariates = static_test[rep(i, each = J),]
    data.s.cur = cbind.data.frame(data.s.cur,  covariates)
  }

  if(p==0){

    #pred.prob = invlogit(predict(model_cur, data.s.cur))
    #pred.prob = invlogit(models_ls[[l]]$coefficients)
    #return(prod(dbinom(data.s.cur$S_dat, 1, pred.prob)))

    pred.prob = models_ls[[l]]
    #apply(data.s.cur, 1, function(x) predict(x, pred.prob))
    pred.prob = invlogit(predict(pred.prob,  data.s.cur))
    return(prod(dbinom(data.s.cur$S_dat, 1, pred.prob)))

  }

  for(j in 1:p){
    lagj = c(rep(NA, j), c(t(s_mat_cur))[1:(J-j)])
    data.s.cur = cbind.data.frame(data.s.cur , lagj)
  }

  data.s.cur = subset(data.s.cur, complete.cases(data.s.cur))

  c1_names=c()
  for(j in 1:p){
    c1_names=c(c1_names, paste("S", j, sep=""))
  }

  names(data.s.cur) = c(names(data.s.cur)[1:(length(names(data.s.cur))-p)],
                        c1_names)


  #get_probs
  pred.prob = invlogit(predict(model_cur, data.s.cur))

  #initial_probs
  initial_combos = as.matrix.data.frame(expand.grid(rep(list(0:1),p)))
  s_mat_test.initial = s_mat_test[i,1:p]
  #which.inits = apply(s_mat_test.initial, 1, function(x) which(apply(initial_combos, 1, function(z) sum(z==x)==p )))
  which.inits = which(apply(initial_combos, 1, function(z) sum(z==s_mat_test.initial)==p ))
  initial_probs.test.mat = initial_probs[which.inits]

  return(prod(c(dbinom(data.s.cur$S_dat, 1, pred.prob), initial_probs.test.mat)))

}






#'Function: Grid search to estimate predicted values and estimate the values of h
#' @param scores N x K matrix of 1st level scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param min.h min possible value for the multiplier
#' @param max.h maximum possible value for the multiplier
#' @param n_grid number of values between min.h and max.h to search over
#' @param CV Number of folds for cross validation
#' @param return_h T/F to return the value of the multiplier
#' @param return_prob T/F to return group Bayes classifier probability for each individual in testing set
#' @param cat_covariates_train Covariates used in the classifier for the training set accounts
#' @param cat_covariates_test Covariates used in the classifier for the testing set accounts
#' @return predictions from the grid search
nb_updated_grid_scores_cat_only = function(scores, cat_covariates_train, classes, prior_g, scores_test, cat_covariates_test,
                                           min.h = 0.01, max.h = 1.5,
                                           CV = 10, n_grid = 10, return_h = F, return_prob = F){

  #create matrix for apply functions
  vec.cv = matrix(1:CV, ncol = 1)

  #create vector of possible CV groups to each account
  #add extra incase unequal distribution of groups
  cvgroups = rep(1:CV, (length(classes)/CV+1))
  #remove the unneeded values
  cvgroups = cvgroups[1:length(classes)]
  #randomly assign CV group to each account
  cvgroups = sample(cvgroups, size = length(classes), replace = F)

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    #If k == 1 change to vector
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  # define function here to use the cvgroups this function
  # will be used in the following apply statement
  get.cv.h = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated_scores_only(scores[cvgroups!=x,], classes[cvgroups!=x],
                                 c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                                 scores[cvgroups==x,], h = h.val) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }

  #initialize matrix for apply which contains all of the possible grid values
  grid.vals.h = matrix(seq(min.h, max.h, length.out = n_grid), ncol = 1)
  #apply the previously defined functions to get the CV accuracies at each h value
  h.accs = apply(grid.vals.h, 1, function(h) get.cv.h(h))

  # assign h value based on the one with the largest CV accuracy
  h = grid.vals.h[which.max(h.accs)]
  #h = grid.vals.h[max(which(h.accs==max(h.accs)))]

  #get K
  nd = dim(scores)[2]
  #get number of groups
  ng = length(unique(classes))
  #initialize list
  densities = list()
  #estimate density at each K for each group
  for(i in 1:ng){
    densities[[i]]=list()
    for(k in 1:nd){
      densities[[i]][[k]] = density(scores[classes==i,k],
                                    kernel = "gaussian",
                                    bw = h*sd(scores[classes==i,k]))
    }
  }

  # get number of test functions
  n_test = dim(scores_test)[1]
  p.mat = matrix(NA, nrow = n_test, ncol = length(prior_g))
  vec = matrix(1:n_test, ncol = 1)
  #For each group get the Bayes classifier value of probability of being in that group
  for(i in 1:ng){
    #apply for each user get the probability for each component in that group
    pdf.vals.test.1 = t(apply(vec, 1,
                              function(x)  get_pdf_den2(densities, scores_test, rep(i, n_test), x)))
    #apply for each user to get the product of those K probabilities
    pdf_vals = apply(pdf.vals.test.1, 1, prod)

    cur_prod = rep(NA, n_test)

    for(p in 1:(dim(cat_covariates_train)[2])){

      cur.levels = unique(c(unlist(cat_covariates_train[,p]), unlist(cat_covariates_test[,p])))
      alpha_new = 5
      cat_covariates_train_cur_group = cat_covariates_train[which(classes==i),]
      cur_cat = table(factor(unlist(cat_covariates_train_cur_group[,p]), cur.levels)) + alpha_new
      cur.var = cur_cat / sum(cur_cat)

      if(p==1){
        cur_prod = unlist(apply(vec, 1, function(x) cur.var[which(unlist(cat_covariates_test[x,p]) == cur.levels)]))
      }else{
        cur_prod = cur_prod * unlist(apply(vec, 1, function(x) cur.var[which(unlist(cat_covariates_test[x,p]) == cur.levels)]))
      }

    }

    #multiply by prior probability
    p.mat[,i] = prior_g[i]* pdf_vals * cur_prod #p.mat is now matrix of Bayes probabilities
  }

  #returns matrix of probabilies for each group
  if(return_prob){
    return(p.mat/rowSums(p.mat))
  }

  #group prediction is based on maximum posterior probability
  guess = apply(p.mat, 1, which.max)

  #print(h)

  if(return_h){
    return(h)
  }

  #return guesses
  return(guess)

}




#' Function: Grid search to estimate predicted values and estimate the values of h
#' @param scores N x K matrix of 1st level scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param min.h min possible value for the multiplier
#' @param max.h maximum possible value for the multiplier
#' @param n_grid number of values between min.h and max.h to search over
#' @param CV Number of folds for cross validation
#' @param s_mat_hat_train Matrix of estimated S data for training
#' @param s_mat_hat_test Matrix of estimated S data for testing
#' @param cat_covariates_train N x Qc data frame of categorical variables
#' @param cat_covariates_test N_new x Qc data frame of categorical variables
#' @param return_h T/F to return the value of the multiplier
#' @param return_prob T/F to return group Bayes classifier probability for each individual in testing set
#' @param P_max number of lags in the gAR model
#' @param static_train Covariates for the classifier in the training set accounts
#' @param static_test Covariates for the classifier in the testing set accounts
#' @param alpha_js Estimated probability of latent states across the realizations
#' @return predictions from the grid search
nb_updated_grid_cat_only = function(scores, s_mat_hat_train, classes, cat_covariates_train, cat_covariates_test,
                           prior_g, scores_test, s_mat_hat_test, alpha_js = NA,
                           min.h = 0.4, max.h = 1.6,
                           CV = 5, n_grid = 5, return_h = F, return_prob = F, P_max = 5,
                           static_train = NA, static_test = NA){

  #create matrix for apply functions
  vec.cv = matrix(1:CV, ncol = 1)


  #create vector of possible CV groups to each account
  #add extra incase unequal distribution of groups
  cvgroups = rep(1:CV, (length(classes)/CV+1))
  #remove the unneeded values
  cvgroups = cvgroups[1:length(classes)]
  #randomly assign CV group to each account
  cvgroups = sample(cvgroups, size = length(classes), replace = F)

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    #If k == 1 change to vector
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  # define function here to use the cvgroups this function
  # will be used in the following apply statement
  get.cv.h.static = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated(scores[cvgroups!=x,], classes[cvgroups!=x],
                     c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                     scores[cvgroups==x,],
                     s_mat_hat_train = s_mat_hat_train[cvgroups!=x,],
                     s_mat_hat_test = s_mat_hat_train[cvgroups==x,],
                     h = h.val, alpha_js = alpha_js, P_max = P_max,
                     static_train = static_train[cvgroups!=x, ],
                     static_test = static_train[cvgroups==x, ]) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }
  get.cv.h = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated(scores[cvgroups!=x,], classes[cvgroups!=x],
                     c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                     scores[cvgroups==x,],
                     s_mat_hat_train = s_mat_hat_train[cvgroups!=x,],
                     s_mat_hat_test = s_mat_hat_train[cvgroups==x,],
                     h = h.val, alpha_js = alpha_js, P_max = P_max) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }

  #initialize matrix for apply which contains all of the possible grid values
  grid.vals.h = seq(min.h, max.h, length.out = n_grid)
  #apply the previously defined functions to get the CV accuracies at each h value
  #if(!is.na(static_train)[1]){
  #  h.accs = sapply(grid.vals.h,
  #                  function(x) get.cv.h.static(x))
  #}else{
    h.accs = sapply(grid.vals.h,
                    function(x) get.cv.h(x))
  #}

  # assign h value based on the one with the largest CV accuracy
  h = h.accs[which.max(h.accs)]
  #h = grid.vals.h[max(which(h.accs==max(h.accs)))]

  guess = nb_updated(scores = scores, classes = classes,
                     prior_g = c(table(classes)/length(classes)),
                     scores_test =  scores_test,
                     s_mat_hat_train = s_mat_hat_train,
                     s_mat_hat_test = s_mat_hat_test, h=h, alpha_js = alpha_js, P_max = P_max,
                     static_train = static_train, static_test = static_test)



  if(return_h){
    return(h)
  }

  #return guesses
  return(guess)

}




#' Function: Grid search to estimate predicted values and estimate the values of h
#' @param scores N x K matrix of 1st level scores in the training set
#' @param classes Group labels vector of length N
#' @param prior_g vector of prior probability of being in each group, sums up to 1
#' @param cat_covariates_train dataframe of categorical variables in the training set
#' @param cat_covariates_test dataframe of categorical variables in the testing set
#' @param scores_test N_test x K matrix of scores in the testing set
#' @param s_mat_hat_train Matrix of estimated S data for training
#' @param s_mat_hat_test Matrix of estimated S data for testing
#' @param min.h min possible value for the multiplier
#' @param max.h maximum possible value for the multiplier
#' @param n_grid number of values between min.h and max.h to search over
#' @param CV Number of folds for cross validation
#' @param return_h T/F to return the value of the multiplier
#' @param return_prob T/F to return group Bayes classifier probability for each individual in testing set
#' @param static_train Covariates for the classifier in the training set accounts
#' @param static_test Covariates for the classifier in the testing set accounts
#' @param P_max number of lags in the gAR model
#' @param alpha_js Estimated probability of latent states across the realizations
#' @return predictions from the grid search
nb_updated_grid_cat = function(scores, s_mat_hat_train, classes, cat_covariates_train, cat_covariates_test,
                               prior_g, scores_test, s_mat_hat_test, alpha_js = NA,
                               min.h = 0.4, max.h = 1.6,
                               CV = 5, n_grid = 5, return_h = F, return_prob = F, P_max = 3,
                               static_train = NA, static_test = NA){

  #create matrix for apply functions
  vec.cv = matrix(1:CV, ncol = 1)


  #create vector of possible CV groups to each account
  #add extra incase unequal distribution of groups
  cvgroups = rep(1:CV, (length(classes)/CV+1))
  #remove the unneeded values
  cvgroups = cvgroups[1:length(classes)]
  #randomly assign CV group to each account
  cvgroups = sample(cvgroups, size = length(classes), replace = F)

  #in case scores are N x 1 matrix that is read as a vector
  if(length(scores)==length(classes)){
    #If k == 1 change to vector
    scores = matrix(scores, ncol = 1)
    scores_test = matrix(scores_test, ncol = 1)
  }

  # define function here to use the cvgroups this function
  # will be used in the following apply statement
  get.cv.h.static = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated(scores[cvgroups!=x,], classes[cvgroups!=x],
                     c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                     scores[cvgroups==x,],
                     s_mat_hat_train = s_mat_hat_train[cvgroups!=x,],
                     s_mat_hat_test = s_mat_hat_train[cvgroups==x,],
                     h = h.val, alpha_js = alpha_js, P_max = P_max,
                     static_train = static_train[cvgroups!=x, ],
                     static_test = static_train[cvgroups==x, ]) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }
  get.cv.h = function(h.val){
    groups.probs =
      apply(vec.cv, 1, function(x)
        mean(#get guess using the nb_updated function
          nb_updated(scores[cvgroups!=x,], classes[cvgroups!=x],
                     c(table(classes[cvgroups!=x])/length(classes[cvgroups!=x])) , #define the new prior probs
                     scores[cvgroups==x,],
                     s_mat_hat_train = s_mat_hat_train[cvgroups!=x,],
                     s_mat_hat_test = s_mat_hat_train[cvgroups==x,],
                     h = h.val, alpha_js = alpha_js, P_max = P_max) ==  classes[cvgroups==x]))
    #return the accuracy of the prediction
    mean(groups.probs)
  }

  #initialize matrix for apply which contains all of the possible grid values
  grid.vals.h = seq(min.h, max.h, length.out = n_grid)
  #apply the previously defined functions to get the CV accuracies at each h value
  if(!is.na(static_train)[1]){
    h.accs = sapply(grid.vals.h,
                    function(x) get.cv.h.static(x))
  }else{
    h.accs = sapply(grid.vals.h,
                    function(x) get.cv.h(x))
  }

  # assign h value based on the one with the largest CV accuracy
  h = h.accs[which.max(h.accs)]
  #h = grid.vals.h[max(which(h.accs==max(h.accs)))]

  # guess = nb_updated(scores = scores, classes = classes,
  #                    prior_g = c(table(classes)/length(classes)),
  #                    scores_test =  scores_test,
  #                    s_mat_hat_train = s_mat_hat_train,
  #                    s_mat_hat_test = s_mat_hat_test, h=h, alpha_js = alpha_js, P_max = P_max,
  #                    static_train = static_train, static_test = static_test)









  #get K
  nd = dim(scores)[2]
  #get number of groups
  ng = length(unique(classes))
  #initialize list
  densities = list()
  #estimate density at each K for each group
  for(i in 1:ng){
    densities[[i]]=list()
    for(k in 1:nd){
      densities[[i]][[k]] = density(scores[classes==i,k],
                                    kernel = "gaussian",
                                    bw = h*sd(scores[classes==i,k]))
    }
  }

  # get number of test functions
  n_test = dim(scores_test)[1]
  p.mat = matrix(NA, nrow = n_test, ncol = length(prior_g))
  vec = matrix(1:n_test, ncol = 1)

  #set ps to 3
  get_ps = rep(P_max, ng)

  models_ls = list()
  p.mat_s = matrix(NA, nrow = n_test, ncol = length(unique(classes)))

  for(l in 1:ng){

    models_ls[[l]] = fit_ajs_model(l, get_ps[l], s_mat_hat_train, classes = classes, static_train = static_train)
    p.mat_s[,l]  = sapply(1:n_test, function(x) predict_new_ind_group_model(x, l, get_ps[l],  models_ls, s_mat_hat_test, static_test))

  }


  #For each group get the Bayes classifier value of probability of being in that group
  for(i in 1:ng){
    #apply for each user get the probability for each component in that group
    pdf.vals.test.1 = t(apply(vec, 1,
                              function(x)  get_pdf_den2(densities, scores_test, rep(i, n_test), x)))
    #apply for each user to get the product of those K probabilities
    pdf_vals = apply(pdf.vals.test.1, 1, prod)
    pdf.vals.test.1 = cbind.data.frame(pdf_vals, p.mat_s[,i])
    pdf_vals = apply(pdf.vals.test.1, 1, prod)

    cur_prod = rep(NA, n_test)

    if(length(cat_covariates_train)>0){

      for(p in 1:(dim(cat_covariates_train)[2])){

        cur.levels = unique(c(unlist(cat_covariates_train[,p]), unlist(cat_covariates_test[,p])))
        alpha_new = 5
        cat_covariates_train_cur_group = cat_covariates_train[which(classes==i),]
        cur_cat = table(factor(unlist(cat_covariates_train_cur_group[,p]), cur.levels)) + alpha_new
        cur.var = cur_cat / sum(cur_cat)

        if(p==1){
          cur_prod = unlist(apply(vec, 1, function(x) cur.var[which(unlist(cat_covariates_test[x,p]) == cur.levels)]))
        }else{
          cur_prod = cur_prod * unlist(apply(vec, 1, function(x) cur.var[which(unlist(cat_covariates_test[x,p]) == cur.levels)]))
        }

      }
      p.mat[,i] = prior_g[i]* pdf_vals * cur_prod #p.mat is now matrix of Bayes probabilities
    }  else{
      p.mat[,i] = prior_g[i]* pdf_vals  #p.mat is now matrix of Bayes probabilities
    }
    #multiply by prior probability

  }

  #returns matrix of probabilies for each group
  if(return_prob){
    return(p.mat/rowSums(p.mat))
  }

  #group prediction is based on maximum posterior probability
  guess = apply(p.mat, 1, which.max)


  if(return_h){
    return(h)
  }

  #return guesses
  return(guess)

}


