source('R/causal.null.test.R')
source('R/mixed.dens.R')

##############################################################

# Set a different library location
lib_location <- "/home/ynychen/.local/lib"
# Set the new library location
.libPaths(lib_location)

# Set the CRAN mirror
#cran_mirror <- "https://cran.r-project.org"

#if (require(SuperLearner) == F){
#  install.packages("SuperLearner", lib = lib_location, repos = cran_mirror)
#}

#if (require(earth) == F){
#  install.packages("earth", lib = lib_location, repos = cran_mirror)
#}

#if (require(randomForest) == F){
#  install.packages("randomForest", lib = lib_location, repos = cran_mirror)
#}

#if (require(Rsolnp) == F){
#  install.packages("Rsolnp", lib = lib_location, repos = cran_mirror)
#}

#if (require(sets) == F){
#  install.packages("sets", lib = lib_location, repos = cran_mirror)
#}

library(SuperLearner, lib = lib_location)
library(earth, lib = lib_location)
#library(randomForest, lib = lib_location)
library(glmnet)
library(Rsolnp, lib = lib_location)
library(sets, lib = lib_location)

##############################################################
num = 200
delta_list = c(0, 0.3, 0.5) #c(0, 0.5, 1) #seq(0, 0.5, 0.1)
data_dir = '/dataset/simu1/eval/'
save_dir = 'R/logs/simu1/eval/'
p = c(1,2,Inf)
#alg_list = c("SL.earth", "SL.glm", "SL.gam", "SL.randomForest")
alg_list = c("SL.earth", "SL.glm", "SL.gam", "SL.glmnet", "SL.mean")
alpha = 0.05
##############################################################

mu.mod <- function(t, x, delta) {
    if (is.null(dim(x))){
        x1 = x[1]
        x3 = x[3]
        x4 = x[4]
        x6 = x[6]
    }else {
        x1 <- x[,1]
        x3 <- x[,3]
        x4 <- x[,4]
    }
    x6 <- x[,6]
    g_t = cos((t - 0.5) * 3.14159 * 2) * delta * t^2 + 1
    y <- cos((t - 0.5) * 3.14159 * 2) * (4 * pmax(x1, x6)^3) / (1 + 2 * x3^2) * sin(x4 - 0.5) + g_t
    return(y)
}

pifunc <- function(a,x){
       x <- as.matrix(x)
       if (is.null(dim(x))){
           x1 = x[1]
           x2 = x[2]
           x3 = x[3]
           x4 = x[4]
           x5 = x[5]
       } else{
           x1 <- x[,1]
           x2 <- x[,2]
           x3 <- x[,3]
           x4 <- x[,4]
           x5 <- x[,5]
       }
       logit.lambda <- (10 * sin(pmax(x1, x2, x3)) + pmax(x3, x4, x5)^3) / (1 + (x1 + x5)^2) + sin(0.5 * x3) * (1 + exp(x4 - 0.5 * x3)) + x3^2 + 2 * sin(x4) + 2 * x5 - 6.5
       return(dnorm(log(a / (1 - a)), mean = logit.lambda, sd = 0.5) * 1/(a * (1-a)))
}


for (delta in delta_list){
    p_val = matrix(rep(0, num * 3), ncol = 3)
    time_cost = rep(0, num)
    p_val2 = matrix(rep(0, num * 3), ncol = 3)
    time_cost2 = rep(0, num)

    mufunc <- function(a,l){
       l <- as.matrix(l)
       return(mu.mod(a,l,delta))
    }

    for (i in 0:(num-1)){
        load_dir = paste(data_dir, toString(i), '/', sep = '')
        dat = read.csv(paste(getwd(), load_dir, 'delta_', toString(delta), '_data.txt', sep = ''), header = F, sep = ' ')

        y = dat[[8]]
        a = dat[[1]]
        l = dat[2:7]

        start_time <- Sys.time()
        out <- causalNullTest(y, a, l, p = p, control = list(mu.hat = mufunc, g.hat = pifunc, cross.fit = FALSE, verbose=FALSE, g.n.bins = 2:5))
        end_time <- Sys.time()
        time_cost[i] = end_time - start_time
        p_val[i,] = out$test$p.val

        start_time <- Sys.time()
        out2 <- causalNullTest(y, a, l, p = p, control = list(mu.SL.library = alg_list, g.SL.library = alg_list, cross.fit = FALSE, verbose=FALSE, g.n.bins = 2:5))
        end_time <- Sys.time()
        time_cost2[i] = end_time - start_time
        p_val2[i,] = out2$test$p.val
    }
    write.table(p_val, file=paste(save_dir, "p_val_w_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(p_val2, file=paste(save_dir, "p_val_w_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost, file=paste(save_dir, "run_time_w_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost2, file=paste(save_dir, "run_time_w_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
}

rej_rate = matrix(rep(0, length(delta_list) * 3), ncol = 3)
run_time = rep(0, length(delta_list))
rej_rate2 = matrix(rep(0, length(delta_list) * 3), ncol = 3)
run_time2 = rep(0, length(delta_list))

idx = 1
for (delta in delta_list){
    out = read.csv(paste(save_dir, "p_val_w_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    out2 = read.csv(paste(save_dir, "p_val_w_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    rej_rate[idx,] = colMeans(out < alpha)
    rej_rate2[idx,] = colMeans(out2 < alpha)

    time_cost_ = read.csv(paste(save_dir, "run_time_w_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    time_cost2_ = read.csv(paste(save_dir, "run_time_w_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    run_time[idx] = mean(time_cost_[[1]])
    run_time2[idx] = mean(time_cost2_[[1]])
    idx = idx + 1
}

result = t(cbind(c('delta', delta_list), rbind(p, rej_rate), c('time', run_time)))
colnames(result) = NULL
result2 = t(cbind(c('delta', delta_list), rbind(p, rej_rate2), c('time', run_time2)))
colnames(result2) = NULL

write.table(result, file=paste(save_dir, 'westling_oracal_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
write.table(result2, file=paste(save_dir, 'westling_SuperLearner_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
