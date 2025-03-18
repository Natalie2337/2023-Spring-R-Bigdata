library(car)
library(foreign)
library(ggplot2)

#donate <- read.csv("r-data_dummy_processed_no_colinearity.csv")
#lm(formula = DAYSWAIT_CHRON ~ ., data = donate)
#lm.donate <- lm(formula = DAYSWAIT_CHRON ~ ., data = donate)
#summary(lm.donate)

donate <- read.csv("r-data_dummy_processed_no_colinearity.csv")
lm.donate <-lm(formula = DAYSWAIT_CHRON ~ IABP_TCR+MALIG_TCR+IMPL_DEFIBRIL+LIFE_SUP_TCR+ADMISSION.RATE+DIAB_3.0+DIAB_5.0, data = donate)
summary(lm.donate)


