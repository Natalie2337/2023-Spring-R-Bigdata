PetSuperStore_sample_data = na.omit(PetSuperStore_sample_data)

###简单观察数据
dim(PetSuperStore_sample_data)
names(PetSuperStore_sample_data)
plot(PetSuperStore_sample_data$PetSuperStore_Price, PetSuperStore_sample_data$cost)
plot(PetSuperStore_sample_data$PetSuperStore_Price, PetSuperStore_sample_data$competitor_price)
summary(PetSuperStore_sample_data)

### Simple Linear Regression

lm.fit=lm(PetSuperStore_Price~cost,data=PetSuperStore_sample_data)
attach(PetSuperStore_sample_data)
lm.fit=lm(PetSuperStore_Price~cost)
lm.fit
summary(lm.fit)
names(lm.fit)
coef(lm.fit)
confint(lm.fit)

predict(lm.fit,data.frame(cost=(c(5,10,15))), interval="confidence")
?predict(lm.fit,data.frame(cost=(c(5,10,15))), interval="prediction")
plot(cost,PetSuperStore_Price)
abline(lm.fit)
abline(lm.fit,lwd=3)
abline(lm.fit,lwd=3,col="red")
plot(cost,PetSuperStore_Price,col="red")
plot(cost,PetSuperStore_Price,pch=20)
plot(cost,PetSuperStore_Price,pch="+")
plot(1:20,1:20,pch=1:20)


par(mfrow=c(2,2))
plot(lm.fit)


plot(predict(lm.fit), residuals(lm.fit))


##### Multiple Linear Regression

lm.fit=lm(ordered_units_t7d~ordered_units_t60d+ordered_units_t90d,data=PetSuperStore_sample_data)
summary(lm.fit)
lm.fit=lm(ordered_units_t7d~.,data=PetSuperStore_sample_data)
summary(lm.fit)
#library(car)
lm.fit1=lm(ordered_units_t7d~.-ordered_units_t12m,data=PetSuperStore_sample_data)
summary(lm.fit1)
lm.fit1=update(lm.fit, ~.-ordered_units_t12m)














