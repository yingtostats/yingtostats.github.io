---
layout: post
title:  "Quasi-Newton"
date:   2016-11-28 21:00:00
tag:
- Code
projects: true
blog: false
author: Jannis
description: Quasi-Newton Methods with comparisons
fontsize: 23pt
---

# Newton型方法的数值比较

## 线搜索程序

程序可以包含精确线搜索准则与不同的非精确性搜索准则以及不同的线搜索求步长的方法。(写得并不好)

### 精确线搜索

#### 进退法求初始搜索区间

* 给定初始值start$\in (0,\infty)$,$r > 0$,$t > 1$, $i:=0$
* alpha.new = start + r, 若alpha.new $\leq 0$,则令alpha.new := 0
* 按照算法，并且限制了最大循环次数max.it
* 返回区间上下限a,b以及循环次数i

```{r}

ls.region <- function(get.alpha,start = 5, r = 0.8,t = 1.5, max.it = 1000) {

    i <- 0
    while(i <= max.it) {
        while(i <= max.it) {
            i <- i + 1
            alpha.new = start + r

            if (alpha.new <= 0) {
                alpha.new = 0
                break
            }else if (get.alpha(alpha.new) >= get.alpha(start)) {
                break
            }

            r <- t*r
            alpha <- start
            start <- alpha.new
        }

        if(i == 1) { #i == 1则需要换方向
            r <- -r
            alpha <- alpha.new
        }else { #i != 1 则完成了初始搜索区间的搜索
            a <- min(alpha,alpha.new)
            b <- max(alpha,alpha.new)
            break
        }

    }
    return(c(a,b,i))
}

```

#### 精确与非精确线搜索程序

* 给定函数f，f的梯度g,$xk = x_{k}$, $dk = d_{k}$
* (a,b)为初始搜索区间的上下限，当线搜索方法为二项插值法时不需要设定。
* exact表示是否使用精确线搜索
* 终止准则是满足criteria准则或者达到精度<precision。
* 在精确搜索时，criteria有三种选择：Goldstein,Wolfe,StrongWolfe, 其中rho 与sigma为其中参数
* 最大循环次数为max = 1000
* method可以为0.618法或者Poly二项插值法（alpha0为初始值）
* 返回一个长度为2的向量，表示步长与循环次数。

```{r}

linesearch <- function(f,g,xk,dk, a = 0.1, b=100, alpha0 = 1, precision = 0.001, tau = 0.618, exact = TRUE, method = "0.618", criteria = "Goldstein",rho = 0.0001, sigma = 0.9, max.it = 1000) {

    get.alpha <- function(alpha) {
      f(xk+alpha*dk)
    }

    i = 1

    if(method != "poly")  {
        while(exact && i <= max.it) { #0.618法的精确搜索

            if((b - a) < precision) { #若b与a相隔很小则直接默认用alpha0
              alpha0 <- (b+a)/2
              break
            }

            alpha.l <- a + (1 - tau) * (b - a)
            alpha.u <- a + tau * (b - a)
            if(get.alpha(alpha.l) - get.alpha(alpha.u) < 0) {
              b <- alpha.u
            }else {
              a <- alpha.l
            }
            i = i + 1

        }

        while(!exact && i<=max.it) { #0.618法非精确搜索

            alpha0 <- (b+a)/2
            fk <- get.alpha(0)
            gd <- t(g(xk))%*%dk
            phi <- get.alpha(alpha0)
            #不同准则
            if( phi <= fk + rho * gd * alpha0) {
                if(criteria == "Goldstein") {
                  if(phi >= fk + (1 - rho) * gd * alpha0) {
                    break
                  }
                }

                if(criteria == "Wolfe") {
                  if(t(g(xk + alpha0*dk))%*%dk >= sigma * gd) {
                    break
                  }
                }          

                if(criteria == "StrongWolfe") {
                  if(abs(t(g(xk + alpha0*dk))%*%dk) <= - sigma * gd) {
                    break
                  }
                }     
            }

            alpha.l <- a + (1 - tau) * (b - a)
            alpha.u <- a + tau * (b - a)

            if(get.alpha(alpha.l) - get.alpha(alpha.u) < 0) {
              b <- alpha.u
            }else {
              a <- alpha.l
            }

            i = i + 1

        }
    }

    if (method == "poly") {#二项插值法

        stopifnot(!exact)#并没有精确搜索的算法

        fk <- get.alpha(0)
        gd <- (t(g(xk))%*%dk)[1,1]

        while(!exact && i<= max.it) {

            phi <- get.alpha(alpha0)

            if(phi <= fk + rho * gd * alpha0) {
                #非精确搜索的三项准则
                if(criteria == "Goldstein") {
                  if((phi >= fk + (1 - rho) * gd * alpha0)&& alpha0 >=0) {
                    break
                  }
                }

                if(criteria == "Wolfe") {
                  if(((t(g(xk + alpha0*dk))%*%dk)[1,1] >= sigma * gd)&& alpha0 >=0) {
                    break
                  }
                }          

                if(criteria == "StrongWolfe") {
                  if((abs(t(g(xk + alpha0*dk))%*%dk)[1,1] <= - sigma * gd) && alpha0 >=0) {
                    break
                  }
                }          
            }

            i <- i + 1
            alpha0 <- (- gd) * alpha0^2 / (2 * (phi - fk - gd * alpha0))
            if(alpha0 < 0) {alpha0 = 0}
        }
    }

    return(c(alpha0,i)) #返回步长与循环次数
}

```

## 阻尼Newton法和修正Newton方法的程序

* 输入函数f,梯度g, hessian矩阵hess,初始值x0,
* method有Newton和Damped(阻尼牛顿法)
* Newton法默认步长alpha为1
* 终止准则是$\parallel g(x) \parallel^{2} < \text{precision}$
* exact表示是否使用精确线搜索。
* ls.method表示线搜索方法
* criteria表示非精确线搜索的准则（配套rho,sigma)
* 区间搜索的参数为r与t
* max为最大循环次数
* 输出trace表示每次更新的值、k表示循环次数，c表示区间搜索总循环次数，p表示线搜索总循环次数

```{r}

Newton <- function(f,g,hess,x0,method = "Newton",precision = 0.00001,exact = FALSE,ls.method = "0.618",criteria = "Goldstein",rho = 0.0001,sigma = 0.1, r = 3, t = 1.1, v = 1,max= 1000) {

    n <- length(x0)
    trace <- matrix(NA,nrow = max, ncol = n)
    trace[1,] <- x0
    k = 1
    c = 0
    p = 0
    alpha = 1

    while(k < max) {

        gk <- g(x0)
        Gk <- hess(x0)
        d <- -solve(Gk)%*%gk

        if (method == "Damped") {

            get.alpha <- function(a1) {f(as.vector(x0+a1*d))[1]}

            if(ls.method == "0.618") {
                region <- ls.region(get.alpha,start = alpha, r = r, t = t,max.it = max)
                a <- region[1]
                b <- region[2]
                c <- c+region[3]
                result.alpha <- linesearch(f,g,x0,d,a = a, b = b,exact = exact,precision = 0.01,criteria = criteria, rho = rho, sigma = sigma,max.it = max/50)
                alpha <- result.alpha[1]
            }else {
                result.alpha <- linesearch(f,g,x0,d,alpha0 = alpha,method = ls.method,precision = 0.01,exact = exact,precision = precision,criteria = criteria,rho = rho, sigma = sigma, max.it = max/50)
                alpha <- result.alpha[1]
            }
            p =  p + result.alpha[2]
        }

        x1 = x0 + alpha*d
        k = k + 1
        trace[k,] <- x1

        if (sum(g(x1)^2) < precision) {
          break
        }
        x0 <- x1
    }

    return(list(trace, c(k,c,p)))
}

```

## SR1方法，BFGS方法，DFP方法的程序

* 函数名为Quasi.Newton
* 输入优化函数f,其梯度g，初始值x0
* method为三种方法中的一种，可以是SR1,BFGS,DFP
* exact表示是否使用精确线搜索。
* ls.method表示线搜索方法
* criteria表示非精确线搜索的准则（配套rho,sigma)
* 区间搜索的参数为r与t
* max为最大循环次数
* 输出trace表示每次更新的值、k表示循环次数，c表示区间搜索总循环次数，p表示线搜索总循环次数

```{r}
Quasi.Newton <-  function(f,g,x0,method = "SR1",precision = 0.00001,exact = FALSE,ls.method = "0.618",criteria = "Goldstein",rho = 0.0001,sigma = 0.1, r = 3, t = 1.1, v = 1,max= 1000) {
    n <- length(x0)
    trace <- matrix(NA,nrow = max, ncol = n)
    trace[1,] <- x0
    k = 1
    c = 0
    p = 0
    Hk = diag(n)
    alpha = 1

    while(k < max) {

        gk <- g(x0)
        d <- -Hk%*%gk
        get.alpha <- function(a1) {f(as.vector(x0+a1*d))[1]}

        if(ls.method == "0.618") {
            region <- ls.region(get.alpha,start = alpha, r = r, t = t,max.it = max)
            a <- region[1]
            b <- region[2]
            c <- c + region[3] #line search region time
            result.alpha <- linesearch(f,g,x0,d,a = a, b = b,exact = exact,precision = 0.01,criteria = criteria,rho = rho, sigma = sigma, max.it = max/50)
            alpha <- result.alpha[1]
        }else {
            result.alpha <- linesearch(f,g,x0,d,alpha0 = alpha,precision = 0.01,method = ls.method,exact = exact,precision = precision,criteria = criteria,rho = rho, sigma = sigma, max.it = max/50)
            alpha <- result.alpha[1]
        }

        p = p + result.alpha[2] #line search times
        x1 = x0 + alpha*d
        k = k + 1
        trace[k,] <- x1
        g1 <- g(x1)

        if (sum(g1^2)< precision) {
          break
        }

        sk <- alpha*d
        yk <- g1 - gk

        if(method == "SR1") {
            Hk <- Hk + ((sk-Hk%*%yk)%*%t(sk-Hk%*%yk))/(t(sk-Hk%*%yk)%*%yk)[1,1]
        }else if (method =="BFGS") {
            Hk <- Hk + (1+t(yk)%*%Hk%*%yk/(t(yk)%*%sk)[1,1])[1,1]*sk%*%t(sk)/(t(yk)%*%sk)[1,1] - (sk%*%t(yk)%*%Hk + Hk%*%yk%*%t(sk))/(t(yk)%*%sk)[1,1]
        }else if (method == "DFP")  {
            Hk <- Hk + (sk%*%t(sk)/(t(sk)%*%yk)[1,1]) - (Hk%*%yk%*%t(yk)%*%Hk/(t(yk)%*%Hk%*%yk)[1,1])
        }

        x0 <- x1
    }
    return(list(trace,c(k,c,p)))
}

```


## 数值实验

### (1)Watson函数

$$r_i(x) = \sum_{j=2}^{n}(j-1)x_{j}t^{j-2}_{i} - (\sum_{i=1}^{n}x_{j}t_{i}^{j-1})^2 - 1$$

其中$t_{i} = i/29$, $1 \leq i\leq 29$, $r_{30}(x) = x_1$, $r_{31}(x) = x_2 - x_{1}^{2} - 1$,$2\leq n \leq 31$,$m = 31$,初始点为$x^{(0)} = (0,0,...,0,0)'$

### (2)Discrete boundary value函数

$$r_{i}(x) = 2x_i -x_{i-1} - x_{i+1} + h^2(x_i + t_i + 1)^{3}/2$$

其中$h = 1/(n+1)$,$t_i = ih$, $x_0 = x_{n+1} = 0$,$m = n$，初始点可选为$x^{(0)} = (t_i (t_{i}-1}),...,t_n (t_{n}-1}))'$

* 设n = 4, 10 ,20, 31分别计算精确线搜索下拟牛顿的三种方法的结果
* 并分别计算在非精确线搜索下的三种准则的结果
* 最后使用收敛结果，函数调用次数，循环次数来衡量其表现，并进行对比
* 牛顿法与阻尼牛顿由于Hessian矩阵出现求逆计算困难，因而无法使用


# Appendix

```{r}

############
##Comparison
############


f <- function(x) {
    n0 <- length(x)
    ti <- (1:29)/29
    r <- rep(0,31)

    r[30] <- x[1]
    r[31] <- x[2] - x[1]^2 - 1

    for ( i in 1:29) {
        r2 <- - (sum(t(x)%*%(ti[i]^(c(1:n0-1)))))^2 - 1
        r1 <- 0

        for ( j in 2:n0) {
            r1 = r1 + (j-1)*x[j]*ti[i]^(j-2)
        }

        r[i] = r1 + r2
    }

    return(sum(r^2))
}

g <- function(x) {
    n0 <- length(x)
    ti <- (1:29)/29

    s <- rep(0,29)
    r <- rep(0,31)
    l <- rep(0,n0)

    g <- rep(0,n0)

    r[30] <- x[1]
    r[31] <- x[2] - x[1]^2 - 1

    for ( i in 1:29) {
        s[i] <- sum(t(x) %*% (ti[i]^(c(1:n0-1))))
        r2 <- - (s[i])^2 - 1
        r1 <- 0

        for ( j in 2:n0) {
            r1 = r1 + (j-1)*x[j]*ti[i]^(j-2)
        }

        r[i] = r1 + r2
    }

    for ( i in 1:29)  {
        for (j in 1:n0) {
            l[j] <- (j-1) * ti[i]^(j-1) - 2 * s[i] * ti[i]^(j-1)
        }
        g <- g + 2*r[i]*l
    }

    g <- g + c(2*x[1],rep(0,n0-1))
    g <- g + 2*(x[2] - x[1]^2 - 1)*c(-2*x[1],1,rep(0,n0-2))
    return(g)
}

hess <- function(x) {
    n0 <- length(x)
    ti <- (1:29)/29
    hessian <- matrix(NA,ncol = n0,nrow = n0)

    for(j in 1:n0) {
        for (i in 1:j)  {
            hessian[i,j] <- -4*sum(ti^(i+j-2))
            hessian[j,i] <- hessian[i,j]
        }
    }

    hessian[1,1] <- 12*x[1]^2-4*x[2]-110
    hessian[1,2] <- -60 - 4*x[1]
    hessian[2,2] <- -4*sum(ti^2) + 2

    return(hessian)
}

#######################################n0 = 4#############################################
n0 = 4
x0 <- rep(0,n0)

#simple.newton<- Newton(f,g,hess,x0,method = "Newton",precision = 0.01,exact = FALSE, criteria = "Goldstein",max = 10000)
#singular matrix! when exact = FALSE/TRUE (either criteria)

#damped <-Newton(f,g,hess,x0,method = "Damped",precision = 0.01,exact = TRUE, criteria = "Goldstein",rho = 0.0001,sigma = 0.9,max = 1000)
#singular matrix!

#exact
sr1.2 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.2.time <- sr1.2[[2]]
sr1.2 <- sr1.2[[1]]
rs.sr1.2 <- sr1.2[!is.na(sr1.2[,1]),]
tail(rs.sr1.2)

#exact
bfgs.2 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.2.time <- bfgs.2[[2]]
bfgs.2 <- bfgs.2[[1]]
rs.bfgs.2 <- sr1.2[!is.na(bfgs.2[,1]),]
tail(rs.bfgs.2)

#exact
dfp.2 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.2.time <- dfp.2[[2]]
dfp.2 <- dfp.2[[1]]
rs.dfp.2 <- dfp.2[!is.na(dfp.2[,1]),]
tail(rs.dfp.2)

#SR1 nonexact
sr1.g <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g.time <- sr1.g[[2]]
sr1.g <- sr1.g[[1]]
rs.sr1.g <- sr1.g[!is.na(sr1.g[,1]),]
tail(rs.sr1.g)

sr1.w <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w.time <- sr1.w[[2]]
sr1.w <- sr1.w[[1]]
rs.sr1.w <- sr1.w[!is.na(sr1.w[,1]),]
tail(rs.sr1.w)

sr1.sw <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw.time <- sr1.sw[[2]]
sr1.sw <- sr1.sw[[1]]
rs.sr1.sw <- sr1.sw[!is.na(sr1.sw[,1]),]
tail(rs.sr1.sw)

#BFGS nonexact
bfgs.g <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g.time <- bfgs.g[[2]]
bfgs.g <- bfgs.g[[1]]
rs.bfgs.g <- bfgs.g[!is.na(bfgs.g[,1]),]
tail(rs.bfgs.g)

bfgs.w <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w.time <- bfgs.w[[2]]
bfgs.w <- bfgs.w[[1]]
rs.bfgs.w <- bfgs.w[!is.na(bfgs.w[,1]),]
tail(rs.bfgs.w)

bfgs.sw <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw.time <- bfgs.sw[[2]]
bfgs.sw <- bfgs.sw[[1]]
rs.bfgs.sw <- bfgs.sw[!is.na(bfgs.sw[,1]),]
tail(rs.bfgs.sw)

#DFP nonexact
dfp.g <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g.time <- dfp.g[[2]]
dfp.g <- dfp.g[[1]]
rs.dfp.g <- dfp.g[!is.na(dfp.g[,1]),]
tail(rs.dfp.g)

dfp.w <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w.time <- dfp.w[[2]]
dfp.w <- dfp.w[[1]]
rs.dfp.w <- dfp.w[!is.na(dfp.w[,1]),]
tail(rs.dfp.w)

dfp.sw <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw.time <- dfp.sw[[2]]
dfp.sw <- dfp.sw[[1]]
rs.dfp.sw <- dfp.sw[!is.na(dfp.sw[,1]),]
tail(rs.dfp.sw)

feva1 <- c(sum(sr1.2.time[2:3]),sum(bfgs.2.time[2:3]),sum(dfp.2.time[2:3]))
exact <- c(sr1.2.time[1],bfgs.2.time[1],dfp.2.time[1])
feva2 <- c(sum(sr1.g.time[2:3]),sum(bfgs.g.time[2:3]),sum(dfp.g.time[2:3]))
gs <- c(sr1.g.time[1],bfgs.g.time[1],dfp.g.time[1])
feva3 <- c(sum(sr1.w.time[2:3]),sum(bfgs.w.time[2:3]),sum(dfp.w.time[2:3]))
wf <- c(sr1.w.time[1],bfgs.w.time[1],dfp.w.time[1])
feva4 <- c(sum(sr1.sw.time[2:3]),sum(bfgs.sw.time[2:3]),sum(dfp.sw.time[2:3]))
swf <- c(sr1.sw.time[1],bfgs.sw.time[1],dfp.sw.time[1])

result <- data.frame(feva1,exact,feva2,gs,feva3,wf,feva4,swf)
rownames(result) <- c("SR1","BFGS","DFP")
colnames(result) <- c("feva","Exact","feva","Goldstein","feva","Wolfe","feva","StrongWolfe")
result
colSums(result)
rowSums(result)

##########################################################n0=10################################################################
n0 = 10
x0 <- rep(0,n0)

#exact
sr1.10 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.10.time <- sr1.10[[2]]
sr1.10 <- sr1.10[[1]]
rs.sr1.10 <- sr1.10[!is.na(sr1.10[,1]),]
tail(rs.sr1.10)
m1 <- f(rs.sr1.10[length(rs.sr1.10[,1]),])

#exact
bfgs.10 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.10.time <- bfgs.10[[2]]
bfgs.10 <- bfgs.10[[1]]
rs.bfgs.10 <- sr1.10[!is.na(bfgs.10[,1]),]
tail(rs.bfgs.10)
m2 <- f(rs.bfgs.10[length(rs.bfgs.10[,1]),])

#exact
dfp.10 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.10.time <- dfp.10[[2]]
dfp.10 <- dfp.10[[1]]
rs.dfp.10 <- dfp.10[!is.na(dfp.10[,1]),]
tail(rs.dfp.10)
m3 <- f(rs.dfp.10[length(rs.dfp.10[,1]),])

#SR1 nonexact
sr1.g10 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g10.time <- sr1.g10[[2]]
sr1.g10 <- sr1.g10[[1]]
rs.sr1.g10 <- sr1.g10[!is.na(sr1.g10[,1]),]
tail(rs.sr1.g10)
m11 <- f(rs.sr1.g10[length(rs.sr1.g10[,1]),])

sr1.w10 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w10.time <- sr1.w10[[2]]
sr1.w10 <- sr1.w10[[1]]
rs.sr1.w10 <- sr1.w10[!is.na(sr1.w10[,1]),]
tail(rs.sr1.w10)
m12 <- f(rs.sr1.w10[length(rs.sr1.w10[,1]),])

sr1.sw10 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw10.time <- sr1.sw10[[2]]
sr1.sw10 <- sr1.sw10[[1]]
rs.sr1.sw10 <- sr1.sw10[!is.na(sr1.sw10[,1]),]
tail(rs.sr1.sw10)
m13 <- f(rs.sr1.sw10[length(rs.sr1.sw10[,1]),])

#BFGS nonexact
bfgs.g10 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g10.time <- bfgs.g10[[2]]
bfgs.g10 <- bfgs.g10[[1]]
rs.bfgs.g10 <- bfgs.g10[!is.na(bfgs.g10[,1]),]
tail(rs.bfgs.g10)
m21 <- f(rs.bfgs.g10[length(rs.bfgs.g10[,1]),])

bfgs.w10 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w10.time <- bfgs.w10[[2]]
bfgs.w10 <- bfgs.w10[[1]]
rs.bfgs.w10 <- bfgs.w10[!is.na(bfgs.w10[,1]),]
tail(rs.bfgs.w10)
m22 <- f(rs.bfgs.w10[length(rs.bfgs.w10[,1]),])

bfgs.sw10 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw10.time <- bfgs.sw10[[2]]
bfgs.sw10 <- bfgs.sw10[[1]]
rs.bfgs.sw10 <- bfgs.sw10[!is.na(bfgs.sw10[,1]),]
tail(rs.bfgs.sw10)
m23 <- f(rs.bfgs.sw10[length(rs.bfgs.sw10[,1]),])

#DFP nonexact
dfp.g10 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g10.time <- dfp.g10[[2]]
dfp.g10 <- dfp.g10[[1]]
rs.dfp.g10 <- dfp.g10[!is.na(dfp.g10[,1]),]
tail(rs.dfp.g10)
m31 <- f(rs.dfp.g10[length(rs.dfp.g10[,1]),])


dfp.w10 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w10.time <- dfp.w10[[2]]
dfp.w10 <- dfp.w10[[1]]
rs.dfp.w10 <- dfp.w10[!is.na(dfp.w10[,1]),]
tail(rs.dfp.w10)
m32 <- f(rs.dfp.w10[length(rs.dfp.w10[,1]),])

dfp.sw10 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw10.time <- dfp.sw10[[2]]
dfp.sw10 <- dfp.sw10[[1]]
rs.dfp.sw10 <- dfp.sw10[!is.na(dfp.sw10[,1]),]
tail(rs.dfp.sw10)
m33 <- f(rs.dfp.sw10[length(rs.dfp.sw10[,1]),])

feva1 <- c(sum(sr1.10.time[2:3]),sum(bfgs.10.time[2:3]),sum(dfp.10.time[2:3]))
feva2 <- c(sum(sr1.g10.time[2:3]),sum(bfgs.g10.time[2:3]),sum(dfp.g10.time[2:3]))
feva3 <- c(sum(sr1.w10.time[2:3]),sum(bfgs.w10.time[2:3]),sum(dfp.w10.time[2:3]))
feva4 <- c(sum(sr1.sw10.time[2:3]),sum(bfgs.sw10.time[2:3]),sum(dfp.sw10.time[2:3]))

f.min.e <- c(m1,m2,m3)
f.min.gs <- c(m11,m21,m31)
f.min.wf <- c(m12,m22,m32)
f.min.sw <- c(m13,m23,m33)
exact <- c(sr1.10.time[1],bfgs.10.time[1],dfp.10.time[1])
gs <- c(sr1.g10.time[1],bfgs.g10.time[1],dfp.g10.time[1])
wf <- c(sr1.w10.time[1],bfgs.w10.time[1],dfp.w10.time[1])
swf <- c(sr1.sw10.time[1],bfgs.sw10.time[1],dfp.sw10.time[1])

result.10 <- data.frame(f.min.e,feva1,exact,f.min.gs,feva2,gs,f.min.wf,feva3,wf,f.min.sw,feva4,swf)
rownames(result.10) <- c("SR1","BFGS","DFP")
colnames(result.10) <- c("min.f","feva","EXACT","min.f","feva","Goldstein","min.f","feva","Wolfe","min.f","feva","StrongWolfe")
result.10
rowSums(result.10[,c(3,6,9,12)])
colSums(result.10[,c(3,6,9,12)])

##########################################################n0=20################################################################
n0 = 20
x0 <- rep(0,n0)

#exact
sr1.20 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.20.time <- sr1.20[[2]]
sr1.20 <- sr1.20[[1]]
rs.sr1.20 <- sr1.20[!is.na(sr1.20[,1]),]
tail(rs.sr1.20)
m1 <- f(rs.sr1.20[length(rs.sr1.20[,1]),])

#exact
bfgs.20 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.20.time <- bfgs.20[[2]]
bfgs.20 <- bfgs.20[[1]]
rs.bfgs.20 <- bfgs.20[!is.na(bfgs.20[,1]),]
tail(rs.bfgs.20)
m2 <- f(rs.bfgs.20[length(rs.bfgs.20[,1]),])

#exact
dfp.20 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.20.time <- dfp.20[[2]]
dfp.20 <- dfp.20[[1]]
rs.dfp.20 <- dfp.20[!is.na(dfp.20[,1]),]
tail(rs.dfp.20)
m3 <- f(rs.dfp.20[length(rs.dfp.20[,1]),])

#SR1 nonexact
sr1.g20 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g20.time <- sr1.g20[[2]]
sr1.g20 <- sr1.g20[[1]]
rs.sr1.g20 <- sr1.g20[!is.na(sr1.g20[,1]),]
tail(rs.sr1.g20)
m11 <- f(rs.sr1.g20[length(rs.sr1.g20[,1]),])

sr1.w20 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w20.time <- sr1.w20[[2]]
sr1.w20 <- sr1.w20[[1]]
rs.sr1.w20 <- sr1.w20[!is.na(sr1.w20[,1]),]
tail(rs.sr1.w20)
m12 <- f(rs.sr1.w20[length(rs.sr1.w20[,1]),])

sr1.sw20 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw20.time <- sr1.sw20[[2]]
sr1.sw20 <- sr1.sw20[[1]]
rs.sr1.sw20 <- sr1.sw20[!is.na(sr1.sw20[,1]),]
tail(rs.sr1.sw20)
m13 <- f(rs.sr1.sw20[length(rs.sr1.sw20[,1]),])

#BFGS nonexact
bfgs.g20 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g20.time <- bfgs.g20[[2]]
bfgs.g20 <- bfgs.g20[[1]]
rs.bfgs.g20 <- bfgs.g20[!is.na(bfgs.g20[,1]),]
tail(rs.bfgs.g20)
m21 <- f(rs.bfgs.g20[length(rs.bfgs.g20[,1]),])

bfgs.w20 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w20.time <- bfgs.w20[[2]]
bfgs.w20 <- bfgs.w20[[1]]
rs.bfgs.w20 <- bfgs.w20[!is.na(bfgs.w20[,1]),]
tail(rs.bfgs.w20)
m22 <- f(rs.bfgs.w20[length(rs.bfgs.w20[,1]),])

bfgs.sw20 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw20.time <- bfgs.sw20[[2]]
bfgs.sw20 <- bfgs.sw20[[1]]
rs.bfgs.sw20 <- bfgs.sw20[!is.na(bfgs.sw20[,1]),]
tail(rs.bfgs.sw20)
m23 <- f(rs.bfgs.sw20[length(rs.bfgs.sw20[,1]),])

#DFP nonexact
dfp.g20 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g20.time <- dfp.g20[[2]]
dfp.g20 <- dfp.g20[[1]]
rs.dfp.g20 <- dfp.g20[!is.na(dfp.g20[,1]),]
tail(rs.dfp.g20)
m31 <- f(rs.dfp.g20[length(rs.dfp.g20[,1]),])


dfp.w20 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w20.time <- dfp.w20[[2]]
dfp.w20 <- dfp.w20[[1]]
rs.dfp.w20 <- dfp.w20[!is.na(dfp.w20[,1]),]
tail(rs.dfp.w20)
m32 <- f(rs.dfp.w20[length(rs.dfp.w20[,1]),])

dfp.sw20 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw20.time <- dfp.sw20[[2]]
dfp.sw20 <- dfp.sw20[[1]]
rs.dfp.sw20 <- dfp.sw20[!is.na(dfp.sw20[,1]),]
tail(rs.dfp.sw20)
m33 <- f(rs.dfp.sw20[length(rs.dfp.sw20[,1]),])


feva1 <- c(sum(sr1.20.time[2:3]),sum(bfgs.20.time[2:3]),sum(dfp.20.time[2:3]))
feva2 <- c(sum(sr1.g20.time[2:3]),sum(bfgs.g20.time[2:3]),sum(dfp.g20.time[2:3]))
feva3 <- c(sum(sr1.w20.time[2:3]),sum(bfgs.w20.time[2:3]),sum(dfp.w20.time[2:3]))
feva4 <- c(sum(sr1.sw20.time[2:3]),sum(bfgs.sw20.time[2:3]),sum(dfp.sw20.time[2:3]))
f.min.e <- c(m1,m2,m3)
f.min.gs <- c(m11,m21,m31)
f.min.wf <- c(m12,m22,m32)
f.min.sw <- c(m13,m23,m33)
exact <- c(sr1.20.time[1],bfgs.20.time[1],dfp.20.time[1])
gs <- c(sr1.g20.time[1],bfgs.g20.time[1],dfp.g20.time[1])
wf <- c(sr1.w20.time[1],bfgs.w20.time[1],dfp.w20.time[1])
swf <- c(sr1.sw20.time[1],bfgs.sw20.time[1],dfp.sw20.time[1])

result.20 <- data.frame(f.min.e,feva1,exact,f.min.gs,feva2,gs,f.min.wf,feva3,wf,f.min.sw,feva4,swf)
rownames(result.20) <- c("SR1","BFGS","DFP")
colnames(result.20) <- c("min.f","feva","EXACT","min.f","feva","Goldstein","min.f","feva","Wolfe","min.f","feva","StrongWolfe")
result.20
rowSums(result.20[,c(3,6,9,12)])
colSums(result.20[,c(3,6,9,12)])

##########################################################n0=31################################################################
n0 = 31
x0 <- rep(0,n0)

#exact
sr1.31 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.31.time <- sr1.31[[2]]
sr1.31 <- sr1.31[[1]]
rs.sr1.31 <- sr1.31[!is.na(sr1.31[,1]),]
tail(rs.sr1.31)
m1 <- f(rs.sr1.31[length(rs.sr1.31[,1]),])

#exact
bfgs.31 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.31.time <- bfgs.31[[2]]
bfgs.31 <- bfgs.31[[1]]
rs.bfgs.31 <- sr1.31[!is.na(bfgs.31[,1]),]
tail(rs.bfgs.31)
m2 <- f(rs.bfgs.31[length(rs.bfgs.31[,1]),])

#exact
dfp.31 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.31.time <- dfp.31[[2]]
dfp.31 <- dfp.31[[1]]
rs.dfp.31 <- dfp.31[!is.na(dfp.31[,1]),]
tail(rs.dfp.31)
m3 <- f(rs.dfp.31[length(rs.dfp.31[,1]),])

#SR1 nonexact
sr1.g31 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g31.time <- sr1.g31[[2]]
sr1.g31 <- sr1.g31[[1]]
rs.sr1.g31 <- sr1.g31[!is.na(sr1.g31[,1]),]
tail(rs.sr1.g31)
m11 <- f(rs.sr1.g31[length(rs.sr1.g31[,1]),])

sr1.w31 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w31.time <- sr1.w31[[2]]
sr1.w31 <- sr1.w31[[1]]
rs.sr1.w31 <- sr1.w31[!is.na(sr1.w31[,1]),]
tail(rs.sr1.w31)
m12 <- f(rs.sr1.w31[length(rs.sr1.w31[,1]),])

sr1.sw31 <- Quasi.Newton(f,g,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw31.time <- sr1.sw31[[2]]
sr1.sw31 <- sr1.sw31[[1]]
rs.sr1.sw31 <- sr1.sw31[!is.na(sr1.sw31[,1]),]
tail(rs.sr1.sw31)
m13 <- f(rs.sr1.sw31[length(rs.sr1.sw31[,1]),])

#BFGS nonexact
bfgs.g31 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g31.time <- bfgs.g31[[2]]
bfgs.g31 <- bfgs.g31[[1]]
rs.bfgs.g31 <- bfgs.g31[!is.na(bfgs.g31[,1]),]
tail(rs.bfgs.g31)
m21 <- f(rs.bfgs.g31[length(rs.bfgs.g31[,1]),])

bfgs.w31 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w31.time <- bfgs.w31[[2]]
bfgs.w31 <- bfgs.w31[[1]]
rs.bfgs.w31 <- bfgs.w31[!is.na(bfgs.w31[,1]),]
tail(rs.bfgs.w31)
m22 <- f(rs.bfgs.w31[length(rs.bfgs.w31[,1]),])

bfgs.sw31 <- Quasi.Newton(f,g,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw31.time <- bfgs.sw31[[2]]
bfgs.sw31 <- bfgs.sw31[[1]]
rs.bfgs.sw31 <- bfgs.sw31[!is.na(bfgs.sw31[,1]),]
tail(rs.bfgs.sw31)
m23 <- f(rs.bfgs.sw31[length(rs.bfgs.sw31[,1]),])

#DFP nonexact
dfp.g31 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g31.time <- dfp.g31[[2]]
dfp.g31 <- dfp.g31[[1]]
rs.dfp.g31 <- dfp.g31[!is.na(dfp.g31[,1]),]
tail(rs.dfp.g31)
m31 <- f(rs.dfp.g31[length(rs.dfp.g31[,1]),])


dfp.w31 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w31.time <- dfp.w31[[2]]
dfp.w31 <- dfp.w31[[1]]
rs.dfp.w31 <- dfp.w31[!is.na(dfp.w31[,1]),]
tail(rs.dfp.w31)
m32 <- f(rs.dfp.w31[length(rs.dfp.w31[,1]),])

dfp.sw31 <- Quasi.Newton(f,g,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw31.time <- dfp.sw31[[2]]
dfp.sw31 <- dfp.sw31[[1]]
rs.dfp.sw31 <- dfp.sw31[!is.na(dfp.sw31[,1]),]
tail(rs.dfp.sw31)
m33 <- f(rs.dfp.sw31[length(rs.dfp.sw31[,1]),])

feva1 <- c(sum(sr1.31.time[2:3]),sum(bfgs.31.time[2:3]),sum(dfp.31.time[2:3]))
feva2 <- c(sum(sr1.g31.time[2:3]),sum(bfgs.g31.time[2:3]),sum(dfp.g31.time[2:3]))
feva3 <- c(sum(sr1.w31.time[2:3]),sum(bfgs.w31.time[2:3]),sum(dfp.w31.time[2:3]))
feva4 <- c(sum(sr1.sw31.time[2:3]),sum(bfgs.sw31.time[2:3]),sum(dfp.sw31.time[2:3]))
f.min.e <- c(m1,m2,m3)
f.min.gs <- c(m11,m21,m31)
f.min.wf <- c(m12,m22,m32)
f.min.sw <- c(m13,m23,m33)
exact <- c(sr1.31.time[1],bfgs.31.time[1],dfp.31.time[1])
gs <- c(sr1.g31.time[1],bfgs.g31.time[1],dfp.g31.time[1])
wf <- c(sr1.w31.time[1],bfgs.w31.time[1],dfp.w31.time[1])
swf <- c(sr1.sw31.time[1],bfgs.sw31.time[1],dfp.sw31.time[1])

result.31 <- data.frame(f.min.e,feva1,exact,f.min.gs,feva2,gs,f.min.wf,feva3,wf,f.min.sw,feva4,swf)
rownames(result.31) <- c("SR1","BFGS","DFP")
colnames(result.31) <- c("min.f","feva","EXACT","min.f","feva","Goldstein","min.f","feva","Wolfe","min.f","feva","StrongWolfe")
result.31
rowSums(result.31[,c(2,4,6,8)])
colSums(result.31[,c(2,4,6,8)])

result.all <- list(result,result.10,result.20,result.31)
save(result.all,file = "experiment1.Rdata")

#############
##Experiment2
#############


f1 <- function(x) {

    n0 <- length(x)
    h <- 1/(1+n0)
    ti <- (1:n0)*h
    X <- c(0,x,0)
    r <- rep(0,n0)

    for (i in 1:n0) {
        r[i] <- 2*X[i+1] - X[i] - X[i+2] + h^2*((X[i+1] + ti[i] + 1)^3)/2
    }
    return(sum(r^2))
}

g1 <- function(x) {

    n0 <- length(x)
    h <- 1/(1+n0)
    ti <- (1:n0)*h
    g <- 3*h^2*(x + ti + 1)^2 + c(2,rep(0,n0-2),2)

    return(g)
}

hess1 <- function(x) {
    n0 <- length(x)
    h <- 1/(1+n0)
    ti <- (1:n0)*h
    return(diag(6*h^2*(x + ti + 1)))
}




#######################################n0 = 4#############################################
n0 <- 4
h <- 1/(1+n0)
ti <- (1:n0)*h
x0 <- (ti*(ti-1))

#simple.newton<- Newton(f,g,hess,x0,method = "Newton",precision = 0.01,exact = FALSE, criteria = "Goldstein",max = 10000)
#singular matrix! when exact = FALSE/TRUE (either criteria)

#damped <-Newton(f,g,hess,x0,method = "Damped",precision = 0.01,exact = TRUE, criteria = "Goldstein",rho = 0.0001,sigma = 0.9,max = 1000)
#singular matrix!

#exact
sr1.2 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.2.time <- sr1.2[[2]]
sr1.2 <- sr1.2[[1]]
rs.sr1.2 <- sr1.2[!is.na(sr1.2[,1]),]
tail(rs.sr1.2)

#exact
bfgs.2 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.2.time <- bfgs.2[[2]]
bfgs.2 <- bfgs.2[[1]]
rs.bfgs.2 <- sr1.2[!is.na(bfgs.2[,1]),]
tail(rs.bfgs.2)

#exact
dfp.2 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.2.time <- dfp.2[[2]]
dfp.2 <- dfp.2[[1]]
rs.dfp.2 <- dfp.2[!is.na(dfp.2[,1]),]
tail(rs.dfp.2)

#SR1 nonexact
sr1.g <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g.time <- sr1.g[[2]]
sr1.g <- sr1.g[[1]]
rs.sr1.g <- sr1.g[!is.na(sr1.g[,1]),]
tail(rs.sr1.g)

sr1.w <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w.time <- sr1.w[[2]]
sr1.w <- sr1.w[[1]]
rs.sr1.w <- sr1.w[!is.na(sr1.w[,1]),]
tail(rs.sr1.w)

sr1.sw <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw.time <- sr1.sw[[2]]
sr1.sw <- sr1.sw[[1]]
rs.sr1.sw <- sr1.sw[!is.na(sr1.sw[,1]),]
tail(rs.sr1.sw)

#BFGS nonexact
bfgs.g <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g.time <- bfgs.g[[2]]
bfgs.g <- bfgs.g[[1]]
rs.bfgs.g <- bfgs.g[!is.na(bfgs.g[,1]),]
tail(rs.bfgs.g)

bfgs.w <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w.time <- bfgs.w[[2]]
bfgs.w <- bfgs.w[[1]]
rs.bfgs.w <- bfgs.w[!is.na(bfgs.w[,1]),]
tail(rs.bfgs.w)

bfgs.sw <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw.time <- bfgs.sw[[2]]
bfgs.sw <- bfgs.sw[[1]]
rs.bfgs.sw <- bfgs.sw[!is.na(bfgs.sw[,1]),]
tail(rs.bfgs.sw)

#DFP nonexact
dfp.g <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g.time <- dfp.g[[2]]
dfp.g <- dfp.g[[1]]
rs.dfp.g <- dfp.g[!is.na(dfp.g[,1]),]
tail(rs.dfp.g)

dfp.w <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w.time <- dfp.w[[2]]
dfp.w <- dfp.w[[1]]
rs.dfp.w <- dfp.w[!is.na(dfp.w[,1]),]
tail(rs.dfp.w)

dfp.sw <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw.time <- dfp.sw[[2]]
dfp.sw <- dfp.sw[[1]]
rs.dfp.sw <- dfp.sw[!is.na(dfp.sw[,1]),]
tail(rs.dfp.sw)

feva1 <- c(sum(sr1.2.time[2:3]),sum(bfgs.2.time[2:3]),sum(dfp.2.time[2:3]))
exact <- c(sr1.2.time[1],bfgs.2.time[1],dfp.2.time[1])
feva2 <- c(sum(sr1.g.time[2:3]),sum(bfgs.g.time[2:3]),sum(dfp.g.time[2:3]))
gs <- c(sr1.g.time[1],bfgs.g.time[1],dfp.g.time[1])
feva3 <- c(sum(sr1.w.time[2:3]),sum(bfgs.w.time[2:3]),sum(dfp.w.time[2:3]))
wf <- c(sr1.w.time[1],bfgs.w.time[1],dfp.w.time[1])
feva4 <- c(sum(sr1.sw.time[2:3]),sum(bfgs.sw.time[2:3]),sum(dfp.sw.time[2:3]))
swf <- c(sr1.sw.time[1],bfgs.sw.time[1],dfp.sw.time[1])

result <- data.frame(feva1,exact,feva2,gs,feva3,wf,feva4,swf)
rownames(result) <- c("SR1","BFGS","DFP")
colnames(result) <- c("feva","Exact","feva","Goldstein","feva","Wolfe","feva","StrongWolfe")
result
colSums(result)
rowSums(result)

##########################################################n0=10################################################################
n0 = 10
h <- 1/(1+n0)
ti <- (1:n0)*h
x0 <- (ti*(ti-1))

#exact
sr1.10 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.10.time <- sr1.10[[2]]
sr1.10 <- sr1.10[[1]]
rs.sr1.10 <- sr1.10[!is.na(sr1.10[,1]),]
tail(rs.sr1.10)
m1 <- f1(rs.sr1.10[length(rs.sr1.10[,1]),])

#exact
bfgs.10 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.10.time <- bfgs.10[[2]]
bfgs.10 <- bfgs.10[[1]]
rs.bfgs.10 <- sr1.10[!is.na(bfgs.10[,1]),]
tail(rs.bfgs.10)
m2 <- f1(rs.bfgs.10[length(rs.bfgs.10[,1]),])

#exact
dfp.10 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.10.time <- dfp.10[[2]]
dfp.10 <- dfp.10[[1]]
rs.dfp.10 <- dfp.10[!is.na(dfp.10[,1]),]
tail(rs.dfp.10)
m3 <- f1(rs.dfp.10[length(rs.dfp.10[,1]),])

#SR1 nonexact
sr1.g10 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g10.time <- sr1.g10[[2]]
sr1.g10 <- sr1.g10[[1]]
rs.sr1.g10 <- sr1.g10[!is.na(sr1.g10[,1]),]
tail(rs.sr1.g10)
m11 <- f1(rs.sr1.g10[length(rs.sr1.g10[,1]),])

sr1.w10 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w10.time <- sr1.w10[[2]]
sr1.w10 <- sr1.w10[[1]]
rs.sr1.w10 <- sr1.w10[!is.na(sr1.w10[,1]),]
tail(rs.sr1.w10)
m12 <- f1(rs.sr1.w10[length(rs.sr1.w10[,1]),])

sr1.sw10 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw10.time <- sr1.sw10[[2]]
sr1.sw10 <- sr1.sw10[[1]]
rs.sr1.sw10 <- sr1.sw10[!is.na(sr1.sw10[,1]),]
tail(rs.sr1.sw10)
m13 <- f1(rs.sr1.sw10[length(rs.sr1.sw10[,1]),])

#BFGS nonexact
bfgs.g10 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g10.time <- bfgs.g10[[2]]
bfgs.g10 <- bfgs.g10[[1]]
rs.bfgs.g10 <- bfgs.g10[!is.na(bfgs.g10[,1]),]
tail(rs.bfgs.g10)
m21 <- f1(rs.bfgs.g10[length(rs.bfgs.g10[,1]),])

bfgs.w10 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w10.time <- bfgs.w10[[2]]
bfgs.w10 <- bfgs.w10[[1]]
rs.bfgs.w10 <- bfgs.w10[!is.na(bfgs.w10[,1]),]
tail(rs.bfgs.w10)
m22 <- f1(rs.bfgs.w10[length(rs.bfgs.w10[,1]),])

bfgs.sw10 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw10.time <- bfgs.sw10[[2]]
bfgs.sw10 <- bfgs.sw10[[1]]
rs.bfgs.sw10 <- bfgs.sw10[!is.na(bfgs.sw10[,1]),]
tail(rs.bfgs.sw10)
m23 <- f1(rs.bfgs.sw10[length(rs.bfgs.sw10[,1]),])

#DFP nonexact
dfp.g10 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g10.time <- dfp.g10[[2]]
dfp.g10 <- dfp.g10[[1]]
rs.dfp.g10 <- dfp.g10[!is.na(dfp.g10[,1]),]
tail(rs.dfp.g10)
m31 <- f1(rs.dfp.g10[length(rs.dfp.g10[,1]),])


dfp.w10 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w10.time <- dfp.w10[[2]]
dfp.w10 <- dfp.w10[[1]]
rs.dfp.w10 <- dfp.w10[!is.na(dfp.w10[,1]),]
tail(rs.dfp.w10)
m32 <- f1(rs.dfp.w10[length(rs.dfp.w10[,1]),])

dfp.sw10 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw10.time <- dfp.sw10[[2]]
dfp.sw10 <- dfp.sw10[[1]]
rs.dfp.sw10 <- dfp.sw10[!is.na(dfp.sw10[,1]),]
tail(rs.dfp.sw10)
m33 <- f1(rs.dfp.sw10[length(rs.dfp.sw10[,1]),])

feva1 <- c(sum(sr1.10.time[2:3]),sum(bfgs.10.time[2:3]),sum(dfp.10.time[2:3]))
feva2 <- c(sum(sr1.g10.time[2:3]),sum(bfgs.g10.time[2:3]),sum(dfp.g10.time[2:3]))
feva3 <- c(sum(sr1.w10.time[2:3]),sum(bfgs.w10.time[2:3]),sum(dfp.w10.time[2:3]))
feva4 <- c(sum(sr1.sw10.time[2:3]),sum(bfgs.sw10.time[2:3]),sum(dfp.sw10.time[2:3]))

f.min.e <- c(m1,m2,m3)
f.min.gs <- c(m11,m21,m31)
f.min.wf <- c(m12,m22,m32)
f.min.sw <- c(m13,m23,m33)
exact <- c(sr1.10.time[1],bfgs.10.time[1],dfp.10.time[1])
gs <- c(sr1.g10.time[1],bfgs.g10.time[1],dfp.g10.time[1])
wf <- c(sr1.w10.time[1],bfgs.w10.time[1],dfp.w10.time[1])
swf <- c(sr1.sw10.time[1],bfgs.sw10.time[1],dfp.sw10.time[1])

result.10 <- data.frame(f.min.e,feva1,exact,f.min.gs,feva2,gs,f.min.wf,feva3,wf,f.min.sw,feva4,swf)
rownames(result.10) <- c("SR1","BFGS","DFP")
colnames(result.10) <- c("min.f","feva","EXACT","min.f","feva","Goldstein","min.f","feva","Wolfe","min.f","feva","StrongWolfe")
result.10
rowSums(result.10[,c(3,6,9,12)])
colSums(result.10[,c(3,6,9,12)])

##############################################################n0=20################################################################
n0 = 20
h <- 1/(1+n0)
ti <- (1:n0)*h
x0 <- (ti*(ti-1))

#exact
sr1.20 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.20.time <- sr1.20[[2]]
sr1.20 <- sr1.20[[1]]
rs.sr1.20 <- sr1.20[!is.na(sr1.20[,1]),]
tail(rs.sr1.20)
m1 <- f1(rs.sr1.20[length(rs.sr1.20[,1]),])

#exact
bfgs.20 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.20.time <- bfgs.20[[2]]
bfgs.20 <- bfgs.20[[1]]
rs.bfgs.20 <- bfgs.20[!is.na(bfgs.20[,1]),]
tail(rs.bfgs.20)
m2 <- f1(rs.bfgs.20[length(rs.bfgs.20[,1]),])

#exact
dfp.20 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.20.time <- dfp.20[[2]]
dfp.20 <- dfp.20[[1]]
rs.dfp.20 <- dfp.20[!is.na(dfp.20[,1]),]
tail(rs.dfp.20)
m3 <- f1(rs.dfp.20[length(rs.dfp.20[,1]),])

#SR1 nonexact
sr1.g20 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g20.time <- sr1.g20[[2]]
sr1.g20 <- sr1.g20[[1]]
rs.sr1.g20 <- sr1.g20[!is.na(sr1.g20[,1]),]
tail(rs.sr1.g20)
m11 <- f1(rs.sr1.g20[length(rs.sr1.g20[,1]),])

sr1.w20 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w20.time <- sr1.w20[[2]]
sr1.w20 <- sr1.w20[[1]]
rs.sr1.w20 <- sr1.w20[!is.na(sr1.w20[,1]),]
tail(rs.sr1.w20)
m12 <- f1(rs.sr1.w20[length(rs.sr1.w20[,1]),])

sr1.sw20 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw20.time <- sr1.sw20[[2]]
sr1.sw20 <- sr1.sw20[[1]]
rs.sr1.sw20 <- sr1.sw20[!is.na(sr1.sw20[,1]),]
tail(rs.sr1.sw20)
m13 <- f1(rs.sr1.sw20[length(rs.sr1.sw20[,1]),])

#BFGS nonexact
bfgs.g20 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g20.time <- bfgs.g20[[2]]
bfgs.g20 <- bfgs.g20[[1]]
rs.bfgs.g20 <- bfgs.g20[!is.na(bfgs.g20[,1]),]
tail(rs.bfgs.g20)
m21 <- f1(rs.bfgs.g20[length(rs.bfgs.g20[,1]),])

bfgs.w20 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w20.time <- bfgs.w20[[2]]
bfgs.w20 <- bfgs.w20[[1]]
rs.bfgs.w20 <- bfgs.w20[!is.na(bfgs.w20[,1]),]
tail(rs.bfgs.w20)
m22 <- f1(rs.bfgs.w20[length(rs.bfgs.w20[,1]),])

bfgs.sw20 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw20.time <- bfgs.sw20[[2]]
bfgs.sw20 <- bfgs.sw20[[1]]
rs.bfgs.sw20 <- bfgs.sw20[!is.na(bfgs.sw20[,1]),]
tail(rs.bfgs.sw20)
m23 <- f1(rs.bfgs.sw20[length(rs.bfgs.sw20[,1]),])

#DFP nonexact
dfp.g20 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g20.time <- dfp.g20[[2]]
dfp.g20 <- dfp.g20[[1]]
rs.dfp.g20 <- dfp.g20[!is.na(dfp.g20[,1]),]
tail(rs.dfp.g20)
m31 <- f1(rs.dfp.g20[length(rs.dfp.g20[,1]),])


dfp.w20 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w20.time <- dfp.w20[[2]]
dfp.w20 <- dfp.w20[[1]]
rs.dfp.w20 <- dfp.w20[!is.na(dfp.w20[,1]),]
tail(rs.dfp.w20)
m32 <- f1(rs.dfp.w20[length(rs.dfp.w20[,1]),])

dfp.sw20 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw20.time <- dfp.sw20[[2]]
dfp.sw20 <- dfp.sw20[[1]]
rs.dfp.sw20 <- dfp.sw20[!is.na(dfp.sw20[,1]),]
tail(rs.dfp.sw20)
m33 <- f1(rs.dfp.sw20[length(rs.dfp.sw20[,1]),])


feva1 <- c(sum(sr1.20.time[2:3]),sum(bfgs.20.time[2:3]),sum(dfp.20.time[2:3]))
feva2 <- c(sum(sr1.g20.time[2:3]),sum(bfgs.g20.time[2:3]),sum(dfp.g20.time[2:3]))
feva3 <- c(sum(sr1.w20.time[2:3]),sum(bfgs.w20.time[2:3]),sum(dfp.w20.time[2:3]))
feva4 <- c(sum(sr1.sw20.time[2:3]),sum(bfgs.sw20.time[2:3]),sum(dfp.sw20.time[2:3]))
f.min.e <- c(m1,m2,m3)
f.min.gs <- c(m11,m21,m31)
f.min.wf <- c(m12,m22,m32)
f.min.sw <- c(m13,m23,m33)
exact <- c(sr1.20.time[1],bfgs.20.time[1],dfp.20.time[1])
gs <- c(sr1.g20.time[1],bfgs.g20.time[1],dfp.g20.time[1])
wf <- c(sr1.w20.time[1],bfgs.w20.time[1],dfp.w20.time[1])
swf <- c(sr1.sw20.time[1],bfgs.sw20.time[1],dfp.sw20.time[1])

result.20 <- data.frame(f.min.e,feva1,exact,f.min.gs,feva2,gs,f.min.wf,feva3,wf,f.min.sw,feva4,swf)
rownames(result.20) <- c("SR1","BFGS","DFP")
colnames(result.20) <- c("min.f","feva","EXACT","min.f","feva","Goldstein","min.f","feva","Wolfe","min.f","feva","StrongWolfe")
result.20
rowSums(result.20[,c(3,6,9,12)])
colSums(result.20[,c(3,6,9,12)])

##########################################################n0=31################################################################
n0 = 31
h <- 1/(1+n0)
ti <- (1:n0)*h
x0 <- (ti*(ti-1))

#exact
sr1.31 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = TRUE, max = 1000)
sr1.31.time <- sr1.31[[2]]
sr1.31 <- sr1.31[[1]]
rs.sr1.31 <- sr1.31[!is.na(sr1.31[,1]),]
tail(rs.sr1.31)
m1 <- f1(rs.sr1.31[length(rs.sr1.31[,1]),])

#exact
bfgs.31 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = TRUE, max = 1000)
bfgs.31.time <- bfgs.31[[2]]
bfgs.31 <- bfgs.31[[1]]
rs.bfgs.31 <- sr1.31[!is.na(bfgs.31[,1]),]
tail(rs.bfgs.31)
m2 <- f1(rs.bfgs.31[length(rs.bfgs.31[,1]),])

#exact
dfp.31 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = TRUE, max = 1000)
dfp.31.time <- dfp.31[[2]]
dfp.31 <- dfp.31[[1]]
rs.dfp.31 <- dfp.31[!is.na(dfp.31[,1]),]
tail(rs.dfp.31)
m3 <- f1(rs.dfp.31[length(rs.dfp.31[,1]),])

#SR1 nonexact
sr1.g31 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5,max=1000)
sr1.g31.time <- sr1.g31[[2]]
sr1.g31 <- sr1.g31[[1]]
rs.sr1.g31 <- sr1.g31[!is.na(sr1.g31[,1]),]
tail(rs.sr1.g31)
m11 <- f1(rs.sr1.g31[length(rs.sr1.g31[,1]),])

sr1.w31 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.w31.time <- sr1.w31[[2]]
sr1.w31 <- sr1.w31[[1]]
rs.sr1.w31 <- sr1.w31[!is.na(sr1.w31[,1]),]
tail(rs.sr1.w31)
m12 <- f1(rs.sr1.w31[length(rs.sr1.w31[,1]),])

sr1.sw31 <- Quasi.Newton(f1,g1,x0,method = "SR1",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
sr1.sw31.time <- sr1.sw31[[2]]
sr1.sw31 <- sr1.sw31[[1]]
rs.sr1.sw31 <- sr1.sw31[!is.na(sr1.sw31[,1]),]
tail(rs.sr1.sw31)
m13 <- f1(rs.sr1.sw31[length(rs.sr1.sw31[,1]),])

#BFGS nonexact
bfgs.g31 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.g31.time <- bfgs.g31[[2]]
bfgs.g31 <- bfgs.g31[[1]]
rs.bfgs.g31 <- bfgs.g31[!is.na(bfgs.g31[,1]),]
tail(rs.bfgs.g31)
m21 <- f1(rs.bfgs.g31[length(rs.bfgs.g31[,1]),])

bfgs.w31 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.w31.time <- bfgs.w31[[2]]
bfgs.w31 <- bfgs.w31[[1]]
rs.bfgs.w31 <- bfgs.w31[!is.na(bfgs.w31[,1]),]
tail(rs.bfgs.w31)
m22 <- f1(rs.bfgs.w31[length(rs.bfgs.w31[,1]),])

bfgs.sw31 <- Quasi.Newton(f1,g1,x0,method = "BFGS",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
bfgs.sw31.time <- bfgs.sw31[[2]]
bfgs.sw31 <- bfgs.sw31[[1]]
rs.bfgs.sw31 <- bfgs.sw31[!is.na(bfgs.sw31[,1]),]
tail(rs.bfgs.sw31)
m23 <- f1(rs.bfgs.sw31[length(rs.bfgs.sw31[,1]),])

#DFP nonexact
dfp.g31 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Goldstein", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.g31.time <- dfp.g31[[2]]
dfp.g31 <- dfp.g31[[1]]
rs.dfp.g31 <- dfp.g31[!is.na(dfp.g31[,1]),]
tail(rs.dfp.g31)
m31 <- f1(rs.dfp.g31[length(rs.dfp.g31[,1]),])


dfp.w31 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "Wolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.w31.time <- dfp.w31[[2]]
dfp.w31 <- dfp.w31[[1]]
rs.dfp.w31 <- dfp.w31[!is.na(dfp.w31[,1]),]
tail(rs.dfp.w31)
m32 <- f1(rs.dfp.w31[length(rs.dfp.w31[,1]),])

dfp.sw31 <- Quasi.Newton(f1,g1,x0,method = "DFP",precision = 0.01, exact = FALSE, criteria = "StrongWolfe", rho = 0.0001,sigma = 0.5, max = 1000)
dfp.sw31.time <- dfp.sw31[[2]]
dfp.sw31 <- dfp.sw31[[1]]
rs.dfp.sw31 <- dfp.sw31[!is.na(dfp.sw31[,1]),]
tail(rs.dfp.sw31)
m33 <- f1(rs.dfp.sw31[length(rs.dfp.sw31[,1]),])

feva1 <- c(sum(sr1.31.time[2:3]),sum(bfgs.31.time[2:3]),sum(dfp.31.time[2:3]))
feva2 <- c(sum(sr1.g31.time[2:3]),sum(bfgs.g31.time[2:3]),sum(dfp.g31.time[2:3]))
feva3 <- c(sum(sr1.w31.time[2:3]),sum(bfgs.w31.time[2:3]),sum(dfp.w31.time[2:3]))
feva4 <- c(sum(sr1.sw31.time[2:3]),sum(bfgs.sw31.time[2:3]),sum(dfp.sw31.time[2:3]))
f.min.e <- c(m1,m2,m3)
f.min.gs <- c(m11,m21,m31)
f.min.wf <- c(m12,m22,m32)
f.min.sw <- c(m13,m23,m33)
exact <- c(sr1.31.time[1],bfgs.31.time[1],dfp.31.time[1])
gs <- c(sr1.g31.time[1],bfgs.g31.time[1],dfp.g31.time[1])
wf <- c(sr1.w31.time[1],bfgs.w31.time[1],dfp.w31.time[1])
swf <- c(sr1.sw31.time[1],bfgs.sw31.time[1],dfp.sw31.time[1])

result.31 <- data.frame(f.min.e,feva1,exact,f.min.gs,feva2,gs,f.min.wf,feva3,wf,f.min.sw,feva4,swf)
rownames(result.31) <- c("SR1","BFGS","DFP")
colnames(result.31) <- c("min.f","feva","EXACT","min.f","feva","Goldstein","min.f","feva","Wolfe","min.f","feva","StrongWolfe")
result.31
rowSums(result.31[,c(2,4,6,8)])
colSums(result.31[,c(2,4,6,8)])

```
