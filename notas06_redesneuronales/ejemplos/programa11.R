#https://cran.r-project.org/web/packages/gbp/vignettes/gbp-vignette.html

library(gbp)

#- it
#  it item <data.table>
#  - oid order id <integer>
#  - sku items id <character>
#  - l it length which scale will be placed along x-coordinate <numeric>
#  - d it depth  which scale will be placed along y-coordinate <numeric>
#  - h it height which scale will be placed along z-coordinate <numeric>
#  - w it weight which scale will be placed along w-coordinate <numeric>
# l d h are subject to rotate, while w is on a separate single dimension
it <- data.table::data.table(
  oid = c(1428571L, 1428571L, 1428571L, 1428572L, 1428572L, 1428572L, 1428572L, 1428572L),
  sku = c("A0A0A0", "A0A0A1", "A0A0A1", "A0A0A0", "A0A0A1", "A0A0A1", "A0A0A2", "A0A0A3"),
  l   = c(2.140000, 7.240000, 7.240000, 2.140000, 7.240000, 7.240000, 6.000000, 4.000000),
  d   = c(3.580000, 7.240000, 7.240000, 3.580000, 7.240000, 7.240000, 6.000000, 4.000000),
  h   = c(4.760000, 2.580000, 2.580000, 4.760000, 2.580000, 2.580000, 6.000000, 4.000000),
  w   = c(243.0000, 110.0000, 110.0000, 243.0000, 110.0000, 110.0000, 235.0000, 258.0000)
)

knitr::kable(it)

#- bn
#  bn bins <data.table>
#  - id bn id <character>
#  - l: bn length limit along x-coordinate <numeric>
#  - d: bn depth  limit along y-coordinate <numeric>
#  - h: bn height limit along z-coordinate <numeric>
#  - w: bn weight limit along w - a separate single dimension <numeric>
#  - l, d, h will be sorted to have l >= d >= h within solver
# bin must be ordered by preference such that the first bin is most preferred one.

bn <- data.table::data.table(
  id = c("K0001", "K0002", "K0003", "K0004", "K0005"),
  l  = c(06.0000, 10.0000, 09.0000, 10.0000, 22.0000),
  d  = c(06.0000, 08.0000, 08.0000, 10.0000, 14.0000),
  h  = c(06.0000, 06.0000, 07.0000, 10.0000, 09.0000),
  w  = c(600.000, 600.000, 800.000, 800.000, 800.000)
)

knitr::kable(bn)

#The function gbp::bpp_solver(it, bn) aims to pack each order into the smallest
#number of bins, and then smallest bins to achieve highest utilization rate,
#subject to the three dimensional none overlap constraints and weight limit
#constraint.

sn <- gbp::bpp_solver(it = it, bn = bn)

#- ldhw: item l, d, h, w in matrix
ldhw <- t(as.matrix(it[oid == 1428572L, .(l, d, h, w)]))
ldhw

#- m: bin l, d, h in matrix
m <- t(as.matrix(bn[ , .(l, d, h, w)])) # multple bin
m

#Resolución
#- p: item fit sequence w.r.t bin
p <- gbp4d_solver_dpp_prep_create_p(ldhw, m[, 4L]) # single bin
p

#- sn
sn4d <- gbp4d_solver_dpp(p, ldhw, m[, 4L])

# matrix of items x, y, z, w (weight bn is holding when fit it into bn),
#l, d, h, w (weight of it itself) (x, y, z, w set into -1 when item not fit
#into bin)
sn4d$it
sn4d$k  # indicator of which items are fitted into bin

gbp4d_viewer(sn4d)
