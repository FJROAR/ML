# vector of impressions per variant
b_Sent<-c(1000, 1000, 100)
b_Sent<-c(1000, 800, 700)

# vector of responses per variant
b_Reward <- c(100, 110, 10)
b_Reward <- c(110, 120, 130)

msgs <- length(b_Sent)

# number of simulations
N <- 5000 

# simulation of Beta distributions (success+1, failures+1)
set.seed(155)
B <- matrix(rbeta(N*msgs, b_Reward+1, (b_Sent-b_Reward)+1),
            N, 
            byrow = TRUE)

# Take the percentage where each variant
# was observed with the highest rate rate
P <- table(factor(max.col(B), levels=1:ncol(B)))/dim(B)[1]
P


