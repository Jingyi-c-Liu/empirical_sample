#Group 19 Jingyi Liu 


###1.Data Pre-processing
library(haven)
library(boot)
library(dplyr)
library(leaps)
library(glmnet)
library(Matrix)
library(Rcpp)
library(pls)
library(ggplot2)
library(splines)
library(tibble)
library(readr)
library(tidyverse)
library(lattice)
library(caret)
library(boot)
library(data.table)
lending<- read_dta('/Users/jingyi/Desktop/机器学习/project/lending club/Data.dta')
#1.1 delete variables not suitable for this prediction 
lending=subset(lending, select = -c(id,zip_code,emp_title,fico_range_high) )
#1.2 delete NA
lending=na.omit(lending)
#1.3 process categorical predictors
lending<-mutate(lending, grade1= case_when(grade == 'A' ~ 1,
                                            grade == 'B'  ~ 2,
                                            grade == 'C'  ~ 3,
                                            grade == 'D'  ~ 4,
                                            grade == 'E'  ~ 5,
                                            grade == 'F'  ~ 6,
                                            grade == 'G'  ~ 7))
lending<-mutate(lending, emp_length1= case_when(emp_length == '< 1 year' ~ 0,
                                                emp_length == '1 year'  ~ 1,
                                                emp_length == '2 years'  ~ 2,
                                                emp_length == '3 years'  ~ 3,
                                                emp_length == '4 years'  ~ 4,
                                                emp_length == '5 years'  ~ 5,
                                                emp_length == '6 years'  ~ 6,
                                                emp_length == '7 years'  ~ 7,
                                                emp_length == '8 years'  ~ 8,
                                                emp_length == '9 years'  ~ 9,
                                                emp_length == '10+ years'  ~ 10))
lending[lending=='ANY']<-NA
lending[lending=='wedding']<-NA
lending=subset(lending, select = -c(grade,emp_length) )
lending=na.omit(lending)

write.csv(lending, file = "lending.csv",row.names = FALSE)
lending<- read.csv("lending.csv", stringsAsFactors = FALSE)

lending$home_ownership<-as.factor(lending$home_ownership)
lending$purpose<-as.factor(lending$purpose)
lending$loan_status<-as.factor(lending$loan_status)
lending$term<-as.factor(lending$term)

###2.Multiple Linear Regression 
#Set training data and test data
set.seed(001)
train = sample(131381,105015)
lending.train <- lending[train, ]
lending.test <- lending[-train, ]

glm.fit=glm(total_acc~.,data=lending.train)
summary(glm.fit)
cv.err=cv.glm(lending.train,glm.fit,K=5)$delta
cv.err #training error is 14.72942 14.72704 
glm.fit=glm(total_acc~.,data=lending)
mean((lending$total_acc-predict(glm.fit,lending))[-train]^2) #test error is 14.9998 



###3.Linear Model Selection and Regularization
#3.1 Best Subset Selection
set.seed(001)
regfit.full=regsubsets(total_acc~.,lending.train, nvmax=29)
reg.summary=summary(regfit.full)
reg.summary
names(reg.summary)
reg.summary$rsq
reg.summary$rss

#3.1.1 choose best predictor numbers using min RSS/cp/bic and max adjr2.
par(mfrow=c(2,2))

plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",
     type="l")
which.min(reg.summary$rss )
points(29,reg.summary$rss [29],col="red",cex=2,pch=20)

plot(reg.summary$adjr2 ,xlab="Number of Variables ",
     ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(23,reg.summary$adjr2[23], col="red",cex=2,pch=20)

plot(reg.summary$cp, xlab="Number of Variables ",ylab="Cp", type='l')
which.min(reg.summary$cp )
points(21,reg.summary$cp [21],col="red",cex=2,pch=20)

plot(reg.summary$bic, xlab="Number of Variables ",ylab="BIC",type='l')
which.min(reg.summary$bic )
points(13,reg.summary$bic [13],col="red",cex=2,pch=20)


#3.1.2 best subset selection with 10-fold
#It leads to 19 predcitors. Best subset selection does not rule out any predictor. 
k=10
folds=sample(1:k,nrow(lending),replace=TRUE)
cv.errors=matrix(NA,k,29, dimnames=list(NULL, paste(1:29)))

predict.regsubsets =function (object ,newdata ,id ,...){
  form=as.formula(object$call [[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object ,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi 
}

for (j in 1:k){
  best.fit=regsubsets(total_acc~.,data=lending[folds!=j,],nvmax=29)
  for (i in 1:29){
    pred=predict(best.fit, lending[folds==j,],id=i)
    cv.errors[j,i]=mean((lending$total_acc[folds==j]-pred)^2)
  }
}

mean.cv.errors=apply(cv.errors ,2,mean) 
mean.cv.errors
par(mfrow=c(1,1))
plot(mean.cv.errors,type='b')
reg.best=regsubsets (total_acc~.,data=lending , nvmax=29)
which.min(mean.cv.errors) 
points(28,mean.cv.errors [28],col="red",cex=2,pch=20)
coef(best.fit,28)
#14.77788 training error
# best subset selection only remove one predictor. 

#one-standard-error 
14.77784+14.77784^0.5
points(4,mean.cv.errors [4],col="blue",cex=2,pch=20)
#15.23368 is within one-standard-error range of the lowest point. 15.23368 only has four predictors.
coef(best.fit,4)
glm.fit=glm(total_acc~open_acc+mort_acc+num_bc_tl+num_il_tl,data=lending.train)
mean((lending$total_acc-predict(glm.fit,lending))[-train]^2)
#test error after one-standard-error is 15.18231.

#3.2 Ridge Regression
x=model.matrix(total_acc~.,lending.train)[,-1] 
y=lending.train$total_acc
set.seed (001)
train=sample(1:nrow(x), nrow(x)/5)
test=(-train)
y.test=y[test]
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam #λ=0.8622253

ridge.mod=glmnet(x,y,alpha=0)
plot(ridge.mod)
ridge.pred=predict(ridge.mod, s=bestlam, newx=x[test,]) 
mean((ridge.pred-y.test)^2)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:29,]
#training error is 15.04306
ridge.mod=glmnet(x,y,alpha=0,lambda=0.8622253 )
x=model.matrix(total_acc~.,lending.test)[,-1] 
mean((lending.test$total_acc-predict(ridge.mod,newx=x))^2)
#test error is 15.40902 with λ=0.8622253.

#3.3 Lasso
x=model.matrix(total_acc~.,lending.train)[,-1] 
y=lending.train$total_acc
set.seed (001)
train=sample(1:nrow(x), nrow(x)/5)
test=(-train)
y.test=y[test]

lasso.mod=glmnet(x[train ,],y[train],alpha=1) 
plot(lasso.mod)

set.seed (001)
cv.out=cv.glmnet(x[train ,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
bestlam #best λ is 0.01857608.
lasso.pred=predict(lasso.mod,s=bestlam ,newx=x[test,])
mean((lasso.pred-y.test)^2) 
out=glmnet(x,y,alpha=1)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:29,] 
lasso.coef
# training error is 14.67124
#11 out of 29 predictors have 0 coefficient. 
lasso.mod=glmnet(x,y,alpha=1,lambda=0.01857608)
x=model.matrix(total_acc~.,lending.test)[,-1] 
mean((lending.test$total_acc-predict(lasso.mod,newx=x))^2)
#testing error is 15.01926 with λ=0.01857608.


#3.4 Principal Components Regression
set.seed (001)
pcr.fit=pcr(total_acc~., data=lending.train ,scale=TRUE,validation ="CV")
summary(pcr.fit)
validationplot(pcr.fit,val.type="MSEP")
#We see that the smallest cross-validation error occurs when M = 28 components are used.MSE=3.837^2=14.72257 
pcr.fit=pcr(total_acc~., data=lending.train,ncomp=28, scale=TRUE,validation ="CV")
mean((lending$total_acc-predict(pcr.fit,lending,ncomp=28))[-train]^2)
#test error is 15.01266 when M = 28.


#3.5 Partial Least Squares
set.seed (001)
pls.fit=plsr(total_acc~., data=lending.train ,scale=TRUE,validation ="CV")
summary(pls.fit)
validationplot(pls.fit,val.type="MSEP")
#We see that the smallest cross-validation error occurs when M = 7 components are used.MSE=3.837^2=14.72257 
pls.fit=plsr(total_acc~., data=lending.train,ncomp=7, scale=TRUE,validation ="CV")
mean((lending$total_acc-predict(pls.fit,lending,ncomp=7))[-train]^2)
#test error is 15.01268 when M = 7.


###4 Moving Beyond Linearity
#4.1 Polynomial
#Residual Plot
glm.fit=glm(total_acc~.,data=lending.train)
predict=predict(glm.fit, newx=lending.train)
residual=lending.train$total_acc-predict(glm.fit, newx=lending.train)
plot(residual,predict)
lm(predict ~ residual)  

#4.1.1 Polynomial with dti
lm(total_acc ~ dti, lending.train)  
ggplot(lending.train, aes(dti, total_acc)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm")
fit.1=lm(total_acc~dti,data=lending.train)
fit.2=lm(total_acc~poly(dti,2),data=lending.train) 
fit.3=lm(total_acc~poly(dti,3),data=lending.train)
fit.4=lm(total_acc~poly(dti,4),data=lending.train) 
fit.5=lm(total_acc~poly(dti,5),data=lending.train) 
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
#A quadratic polynomial appears to provide a reasonable fit to the data, but lower- or higher-order models are not justified.  

#4.1.2 Polynomial with loan_amnt
lm(total_acc ~ loan_amnt, lending.train)  
ggplot(lending.train, aes(loan_amnt, total_acc)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm")
fit.1=lm(total_acc~loan_amnt,data=lending.train)
fit.2=lm(total_acc~poly(loan_amnt,2),data=lending.train) 
fit.3=lm(total_acc~poly(loan_amnt,3),data=lending.train)
fit.4=lm(total_acc~poly(loan_amnt,4),data=lending.train) 
fit.5=lm(total_acc~poly(loan_amnt,5),data=lending.train) 
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
#A fifth power polynomial appear to provide a reasonable fit to the data, but lower-order models are not justified.
 
#4.1.3 Polynomial with fico_range_low
lm(total_acc ~ fico_range_low, lending.train)
ggplot(lending.train, aes(fico_range_low, total_acc)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm")
fit.1=lm(total_acc~fico_range_low,data=lending.train)
fit.2=lm(total_acc~poly(fico_range_low,2),data=lending.train) 
fit.3=lm(total_acc~poly(fico_range_low,3),data=lending.train)
fit.4=lm(total_acc~poly(fico_range_low,4),data=lending.train) 
fit.5=lm(total_acc~poly(fico_range_low,5),data=lending.train) 
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
#A cubic power polynomial appear to provide a reasonable fit to the data, but lower-order models are not justified.


#4.2 Step Functions
#4.2.1 Step Functions with loan_amnt
table(cut(lending.train$loan_amnt,4))
fit=lm(total_acc~cut(lending.train$loan_amnt,4),data=lending.train)
coef(summary(fit))

#4.2.2 Step Functions with dti
table(cut(lending.train$dti,4))
fit=lm(total_acc~cut(lending.train$dti,4),data=lending.train)
coef(summary(fit))

#4.2.2 Step Functions with fico_range_low
table(cut(lending.train$fico_range_low,4))
fit=lm(total_acc~cut(lending.train$fico_range_low,4),data=lending.train)
coef(summary(fit))

#4.3 Splines
set.seed(001)
library(splines)
fico_range_lowlims=range(lending.train$fico_range_low)
fico_range_low.grid=seq(from=fico_range_lowlims[1],to=fico_range_lowlims[2])
fit=lm(total_acc~bs(fico_range_low,),data=lending.train)
summary(fit)
#We fit regression splines with dfs between 3 and 21. Train RSS monotonically decreases till df=16.
all.cv = rep(NA, 21)
for (i in 3:21) {
  lm.fit = lm(total_acc~bs(fico_range_low, df=i), data=lending.train)
  all.cv[i] = sum(lm.fit$residuals^2)
}
all.cv[-c(1, 2)]

#Finally, we use a 5-fold cross validation to find best df. We try all integer values of df between 3 and 21.CV error is more jumpy in this case, but attains minimum at df=12. We pick $12$ as the optimal degrees of freedom.
all.cv = rep(NA, 21)
for (i in 3:21) {
  lm.fit = glm(total_acc~bs(fico_range_low, df=i), data=lending.train)
  all.cv[i] = cv.glm(lending.train, lm.fit, K=5)$delta[2]
}
plot(3:21, all.cv[-c(1, 2)], lwd=2, type="l", xlab="df", ylab="CV error")

fit=lm(total_acc~bs(fico_range_low,df=12),data=lending.train)
pred=predict(fit,newdata=list(fico_range_low=fico_range_low.grid),se=T)
plot(lending.train$fico_range_low,lending.train$total_acc,col="gray")
lines(fico_range_low.grid,pred$fit,lwd=2)
lines(fico_range_low.grid,pred$fit+2*pred$se,lty="dashed")
lines(fico_range_low.grid,pred$fit-2*pred$se,lty="dashed")
dim(bs(fico_range_low.grid,df=12))
attr(bs(fico_range_low.grid,df=12),"knots")
pred.bs = predict(fit, lending.test)
spl.err =mean((lending.test$total_acc- pred.bs)^2)

#Natural Spline
set.seed(001)
fit2=lm(total_acc~ns(fico_range_low,df=12),data=lending.train)
pred2=predict(fit2,newdata=list(fico_range_low=fico_range_low.grid),se=T)
lines(fico_range_low.grid, pred2$fit,col="red",lwd=2)
pred.ns = predict(fit2, lending.test)
ns.err =mean((lending.test$total_acc- pred.ns)^2)

#Smoothing Spline
set.seed(001)
plot(lending.train$fico_range_low,lending.train$total_acc,xlim=fico_range_lowlims,cex=.5,col="darkgrey")
title("Smoothing Spline")
fit=smooth.spline(lending.train$fico_range_low,lending.train$total_acc,df=16)
fit2=smooth.spline(lending.train$fico_range_low,lending.train$total_acc,cv=F)
fit2$df
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF","9.98 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)
pred.s = predict(fit2, lending.test)
ss.err =mean((lending.test$total_acc- pred.s)^2)

#4.4 Generalized additive models
set.seed(001)
library(foreach)
library(gam)
library(leaps)
reg.fit = regsubsets(total_acc ~ ., data = lending.train, nvmax = 19, method = "forward")
reg.summary = summary(reg.fit)
par(mfrow = c(1, 3))
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
min.cp = min(reg.summary$cp)
which.min(reg.summary$cp)
std.cp = sd(reg.summary$cp)
abline(h = min.cp-0.05*std.cp, col = "red", lty = 2)
abline(h = min.cp+0.05*std.cp, col = "red", lty = 2)
plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
min.bic = min(reg.summary$bic)
which.min(reg.summary$bic)
std.bic = sd(reg.summary$bic)
abline(h = min.bic-0.05*std.bic, col = "red", lty = 2)
abline(h = min.bic+0.05*std.bic, col = "red", lty = 2)
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", 
     type = "l")
min.rss = min(reg.summary$rss)
which.min(reg.summary$rss)
std.rss = sd(reg.summary$rss)
abline(h = min.rss-0.05*std.rss, col = "red", lty = 2)
abline(h = min.rss+0.05*std.rss, col = "red", lty = 2)

coefi = coef(reg.fit, id = 4)
names(coefi)

set.seed(001)
gam.fit = gam(total_acc~s(num_il_tl, df = 4)+ s(open_acc, df = 4) + s(mort_acc, df = 4) + s(num_bc_tl, df = 4), data = lending.train)
par(mfrow = c(2, 2))
plot(gam.fit, se = T, col = "blue")

gam.pred = predict(gam.fit, lending.test)
gam.err = mean((lending.test$total_acc - gam.pred)^2)
gam.err


###5 Regression tree
library(tree)
library(MASS)
set.seed(001)

tree.lending = tree(total_acc~., data = lending.train)
summary(tree.lending)
plot(tree.lending)
text(tree.lending,pretty=0)
pred.lending = predict(tree.lending, lending.test)
mean((lending.test$total_acc- pred.lending)^2)

cv.lending=cv.tree(tree.lending, FUN = prune.tree)
par(mfrow = c(1, 2))
plot(cv.lending$size,cv.lending$dev,type='b')
plot(cv.lending$k,cv.lending$dev,type='b')

pruned.lending=prune.tree(tree.lending,best=9)
par(mfrow = c(1, 1))
plot(pruned.lending)
text(pruned.lending,pretty=0)
pred.pruned = predict(pruned.lending,lending.test)
mean((lending.test$total_acc - pred.pruned)^2)

#5.1 Bagging
library(randomForest)
set.seed(001)
bag.lending = randomForest(total_acc ~ ., data = lending.train, mtry = 18, importance = T)
bag.pred = predict(bag.lending, lending.test)
mean((lending.test$total_acc - bag.pred)^2)
importance(bag.lending)

#5.2 Random Forest
set.seed(001)
rf.lending = randomForest(total_acc ~ ., data = lending.train, mtry = 6, importance = T)
rf.pred = predict(rf.lending, lending.test)
mean((lending.test$total_acc - rf.pred)^2)
importance(rf.lending)
varImpPlot (rf.lending)

#5.3 Boosting
install.packages("gbm")
library(gbm)
set.seed(001)
boost.lending=gbm(total_acc~.,data=lending.train,distribution="gaussian",n.trees=5000,interaction.depth=1)
summary(boost.lending)
yhat.boost=predict(boost.lending,newdata=lending.test,n.trees=5000)
mean((yhat.boost-lending.test$total_acc)^2)











































