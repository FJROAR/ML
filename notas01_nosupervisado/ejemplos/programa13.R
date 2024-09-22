#https://cran.r-project.org/web/packages/dbscan/vignettes/hdbscan.html

library("dbscan")

data("moons")
plot(moons, pch=20)

cl <- hdbscan(moons, minPts = 5)
cl

plot(moons, col=cl$cluster+1, pch=20)

#The resulting HDBSCAN object contains a hierarchical representation of every 
#possible DBSCAN* clustering. This hierarchical representation is compactly 
#stored in the familiar ‘hc’ member of the resulting HDBSCAN object, in the same 
#format of traditional hierarchical clustering objects formed using the ‘hclust’ 
#method from the stats package.

cl$hc

plot(cl$hc, main="HDBSCAN* Hierarchy")

###############################################################################
#                                                                             #  
#                           UN EJEMPLO GRANDE                                 #
#                                                                             #  
###############################################################################

data("DS3")
plot(DS3, pch=20, cex=0.25)

cl2 <- hdbscan(DS3, minPts = 25)
cl2

plot(DS3, col=cl2$cluster+1, 
     pch=ifelse(cl2$cluster == 0, 8, 1), # Mark noise as star
     cex=ifelse(cl2$cluster == 0, 0.5, 0.75), # Decrease size of noise
     xlab=NA, ylab=NA)
colors <- sapply(1:length(cl2$cluster), 
                 function(i) adjustcolor(palette()[(cl2$cluster+1)[i]], alpha.f = cl2$membership_prob[i]))
points(DS3, col=colors, pch=20)


plot(cl2, scale = 3, gradient = c("purple", "orange", "red"), show_flat = T)