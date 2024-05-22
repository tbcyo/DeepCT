# Set a different library location
lib_location <- "/home/ynychen/.local/lib"
#lib_location = "~/R/x86_64-pc-linux-gnu-library/4.2"
## Set the new library location
.libPaths(lib_location)

# Set the CRAN mirror
#cran_mirror <- "https://cran.r-project.org"


#if (require(DRDRtest) == F){
#  install.packages("DRDRtest", lib = lib_location, repos = cran_mirror)
#}

#if (require(SuperLearner) == F){
#  install.packages("SuperLearner", lib = lib_location, repos = cran_mirror)
#}

#if (require(earth) == F){
#  install.packages("earth", lib = lib_location, repos = cran_mirror)
#}

#if (require(glmnet) == F){
#  install.packages("earth", lib = lib_location, repos = cran_mirror)
#}

#if (require(randomForest) == F){
#  install.packages("randomForest", lib = lib_location, repos = cran_mirror)
#}

library(DRDRtest, lib = lib_location)
library(SuperLearner) #, lib = lib_location)
library(earth) #, lib = lib_location)
library(glmnet) #randomForest, lib = lib_location)

##############################################################
num = 200
b = 1000
delta_list = c(0, 0.3, 0.5) #c(0, 0.5, 1) #seq(0, 0.5, 0.1)
data_dir = '/dataset/simu1/eval/'
save_dir = 'R/logs/simu1/eval/'
# default algs: "SL.earth", "SL.glm", "SL.gam", "SL.glmnet"
#alg_list = c("SL.mean", "SL.earth", "SL.glm", "SL.gam") # "SL.randomForest")
alg_list = c("SL.earth", "SL.glm", "SL.gam", "SL.glmnet", "SL.mean")
alpha = 0.05
##############################################################
if (dir.exists(substr(save_dir, 1, 1)) == F){
    dir.create(substr(save_dir, 1, 1))
}
if (dir.exists(substr(save_dir, 1, 6)) == F){
    dir.create(substr(save_dir, 1, 6))
}
if (dir.exists(substr(save_dir, 1, 12)) == F){
    dir.create(substr(save_dir, 1, 12))
}
if (dir.exists(save_dir) == F){
    dir.create(save_dir)
}

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
    p_val = rep(0, num)
    p_val2 = rep(0, num)
    time_cost = rep(0, num)
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
      out <- drdrtest(y, a, l, c(0.01,0.99), pifunc, mufunc, b = b)
      end_time <- Sys.time()
      time_cost[i] = end_time - start_time
      p_val[i] = out$p.value

      start_time <- Sys.time()
      out <- drdrtest.superlearner(y, a, l, c(0.01,0.99), pi.sl.lib = alg_list, mu.sl.lib = alg_list, b = b)
      end_time <- Sys.time()
      time_cost2[i] = end_time - start_time
      p_val2[i] = out$p.value
    }

    write.table(p_val, file=paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(p_val2, file=paste(save_dir, "p_val_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost, file=paste(save_dir, "run_time_oracal_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
    write.table(time_cost2, file=paste(save_dir, "run_time_SuperLearner_delta_", delta , '.txt', sep = ''), row.names=FALSE, col.names=FALSE)
}


rej_rate = rep(0, length(delta_list))
run_time = rep(0, length(delta_list))
rej_rate2 = rep(0, length(delta_list))
run_time2 = rep(0, length(delta_list))

idx = 1
for (delta in delta_list){
    out = read.csv(paste(save_dir, "p_val_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    out2 = read.csv(paste(save_dir, "p_val_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    rej_rate[idx] = mean(out < alpha)
    rej_rate2[idx] = mean(out2 < alpha)

    time_cost = read.csv(paste(save_dir, "run_time_oracal_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    time_cost2 = read.csv(paste(save_dir, "run_time_SuperLearner_delta_", delta , '.txt', sep = ''), header  = F, sep = ' ')
    run_time[idx] = mean(time_cost[[1]])
    run_time2[idx] = mean(time_cost2[[1]])
    idx = idx + 1
}
write.table(rbind(delta_list, rej_rate, run_time), file=paste(save_dir, 'oracal_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
write.table(rbind(delta_list, rej_rate2, run_time2), file=paste(save_dir, 'SuperLearner_rej_rate.txt', sep = ''), row.names=FALSE, col.names=FALSE)
