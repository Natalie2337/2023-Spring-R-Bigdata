library(foreign)
library(haven)
library(ggplot2)

Pet <- read.csv("Processed01.csv")
summary(Pet)

#检查相关关系
cor(Pet[, c("image_exists", "brand_name_exists", "product_data_exists")])
cor(Pet[, c("PetSuperStore_Price", "brand_name_exists", "ordered_units_t7d")])
##部分品牌影响订购量
ols.advertising = lm(ordered_units_t12m~brand_name,data = Pet)
summary(ols.advertising)
##包装印有品牌名影响订购量
ols.advertising = lm(ordered_units_t12m~image_exists+brand_name_exists+product_data_exists,data = Pet)
summary(ols.advertising)
##价格影响订购量
ols.advertising = lm(ordered_units_t12m~PetSuperStore_Price,data = Pet)
summary(ols.advertising)
##历史数据影响订购量
ols.advertising = lm(ordered_units_t12m~ordered_units_t7d,data = Pet)
summary(ols.advertising)

#尝试建立一个线性回归模型用来预测用户的采购量
# 分割训练集和测试集
set.seed(123)
train_idx <- sample(1:nrow(Pet), nrow(Pet)*0.7, replace = FALSE)
train_data <- Pet[train_idx, ]
test_data <- Pet[-train_idx, ]

# 标准化处理
train_data_scaled <- train_data
test_data_scaled <- test_data

train_data_scaled[, c("PetSuperStore_Price", "brand_name_exists", "ordered_units_t7d")] <- scale(train_data_scaled[, c("PetSuperStore_Price", "brand_name_exists", "ordered_units_t7d")])
test_data_scaled[, c("PetSuperStore_Price", "brand_name_exists", "ordered_units_t7d")] <- scale(test_data_scaled[, c("PetSuperStore_Price", "brand_name_exists", "ordered_units_t7d")])

# 线性回归模型训练
lm.fit <- lm(ordered_units_t12m ~ PetSuperStore_Price + brand_name_exists + ordered_units_t7d , data = train_data_scaled)

# 查看模型摘要
summary(lm.fit)

# 测试集预测
predicted <- predict(lm.fit, newdata = test_data_scaled)
actual <- test_data$ordered_units_t12m

# 计算R平方和均方误差
R2 <- cor(predicted, actual)^2
mse <- mean((predicted - actual)^2)

# 查看R平方和均方误差
cat(sprintf("R平方为 %f，均方误差为 %f", R2, mse))

#解释：
#PetSuperStore_Price: 相关系数为-18.14，说明价格每增加1单位，平均销售额将减少18.14单位，但是p值为0.2312，因此该变量不显著。
#brand_name_exists: 相关系数为25.01，说明使用品牌名字的商品平均销售额要高于不使用品牌名字的商品。然而，p值为0.0989，因此该变量在5%的显著性水平下不显著。
#ordered_units_t7d: 相关系数为6122.10，说明每增加7天的销售量，平均销售额将增加6122.10单位，结果显著
#F值为54530，p值小于0.05，说明模型整体显著
#总的来说，这个模型整体表现不错，拟合程度较高，但是其中一个自变量不显著，可能需要重新选择变量来提高模型预测精度。



#####尝试
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