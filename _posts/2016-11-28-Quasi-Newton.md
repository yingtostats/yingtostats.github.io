---
layout: post
title:  "Quasi-Newton"
date:  2016-11-28 21:00:00
tag:
- 中文
- Code
projects: true
blog: false
author: Jannis
description: Quasi-Newton Methods with comparisons
---

{% include mathjax_support.html %}

# Newton型方法的数值比较

## 线搜索程序

程序可以包含精确线搜索准则与不同的非精确性搜索准则以及不同的线搜索求步长的方法。(写得并不好)

### 精确线搜索

#### 进退法求初始搜索区间

* 给定初始值start$\in (0,\infty)$,$r > 0$,$t > 1$, $i:=0$
* alpha.new = start + r, 若alpha.new $\leq 0$,则令alpha.new := 0
* 按照算法，并且限制了最大循环次数max.it
* 返回区间上下限a,b以及循环次数i

```

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

```

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


```

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

```

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

```

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

```
