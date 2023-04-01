# clear plots
if(!is.null(dev.list())) dev.off()

# clear console
cat("\014")

# clean workspace
rm(list=ls())

setwd("C:/Users/Checo/Desktop/projectum/github_repos/2021/emailMarketing")
getwd()

## 1. Read train and validation datasets
options(scipen=10)
train = readRDS(paste0("C:/Users/Checo/Desktop/projectum/github_repos/2021/emailMarketing/train.rda"))
valid = readRDS(paste0("C:/Users/Checo/Desktop/projectum/github_repos/2021/emailMarketing/valid.rda"))
dim(train)
dim(valid)

## 2. Information value (IV)

# Remove all variables with an IV less than 0.05 and create a new training and validation data sets
#install.packages("Information")
library(Information)
IV = create_infotables(data=train, y ="PURCHASE", ncore=2)
#View(IV$Summary)

train_new = train[,c(subset(IV$Summary,IV>0.05)$Variable, "PURCHASE")]
dim(train_new)

valid_new=valid[,c(subset(IV$Summary, IV>0.05)$Variable, "PURCHASE")]
dim(valid_new)

## 3. Eliminate highly correlated variables using variable clustering (ClustOfVar package)
# Select the most informative 20 variables to be used for the classification using Variable Clustering

# install.packages("ClustOfVar")
# install.packages("reshape2")
# install.packages("plyr")
library(ClustOfVar)
library(reshape2)
library(plyr)

tree = hclustvar(train_new[,!(names(train_new) == "PURCHASE")])
nvars = 20
part_init = cutreevar(tree,nvars)$cluster
kmeans = kmeansvar(X.quanti=train_new[,!(names(train_new)=="PURCHASE")],init=part_init)
clusters = cbind.data.frame(melt(kmeans$cluster), row.names(melt(kmeans$cluster)))
names(clusters) = c("Cluster", "Variable")
clusters = join(clusters, IV$Summary, by="Variable", type="left")
clusters = clusters[order(clusters$Cluster),]
clusters$Rank = ave(-clusters$IV, clusters$Cluster, FUN=rank)
#View(clusters)
variables = as.character(subset(clusters, Rank==1)$Variable)
variables # Final 20 variables that will be used for classification purporses.

## 4. 
# Create a new response variable called "NEWPurchase" using "PURCHASE" variable in the train data set and add it to the train set
new_purchase_train = c(train_new$PURCHASE)
new_purchase_train = ifelse(train_new$PURCHASE == 1 , 1  , -1)
#new_purchase_train
cbind(train_new, new_purchase_train)
mytraindata= cbind(train_new[variables],new_purchase_train)
str(mytraindata) 
# Change numeric values of predictors to factor 
mytraindata$D_REGION_A = as.factor(mytraindata$D_REGION_A)
mytraindata$D_N_DISPUTED_ACTS = as.factor(mytraindata$D_N_DISPUTED_ACTS)
mytraindata$new_purchase_train = as.factor(mytraindata$new_purchase_train)

# Create a new response variable called "NEWPurchase" using "PURCHASE" variable in the validation data set and add it to the validation set
new_purchase_test = c(valid_new$PURCHASE)
new_purchase_test = ifelse(valid_new$PURCHASE == 1 , 1  , -1)
# new_purchase_test
cbind(valid_new, new_purchase_test)
mytestdata= cbind(valid_new[variables],new_purchase_test)
# mytestdata
str(mytestdata)
# Change numeric predictors to factors
mytestdata$D_REGION_A = as.factor(mytestdata$D_REGION_A)
mytestdata$D_N_DISPUTED_ACTS = as.factor(mytestdata$D_N_DISPUTED_ACTS)
mytestdata$new_purchase_test = as.factor(mytestdata$new_purchase_test)

## 5.  Random Forest
# 5.1 Build up a Random forest using 1,001 trees. Use different "mtry" values varying from 1 to 13. Evaluate the OOB error for each model
library(randomForest)
set.seed(123)
oob.values = vector(length=13)
for(i in 1:13) {
  temp.model = randomForest(new_purchase_train ~ ., data=mytraindata, mtry=i, ntree=1001, importance = TRUE)
  oob.values[i] = temp.model$err.rate[nrow(temp.model$err.rate),1]
}
temp.model
plot(temp.model)
oob.values
#Find minimum error
min(oob.values)   
# Find the optimal value for mtry
which(oob.values == min(oob.values))    

# 5.2 Use model with lowest OBB error and create a variable importance plot
set.seed(123)
best_model = randomForest(new_purchase_train ~ . , data = mytraindata, mtry = 10, ntree = 1001) 
best_model
plot(best_model)
importance(best_model)
order(importance(best_model))
varImpPlot(best_model)
varImpPlot(best_model,pch=18,col="red",cex=0.8)

# 5.3 Prediction
set.seed(123)
predict_model = data.frame(mytestdata$new_purchase_test, predict(best_model,mytestdata,type="response"))
predict_model
plot(predict_model)

# 5.4 Evaluate the confusion matrix table and calculate the sensitivity,specificity and accuracy
library(caret)
library(ggplot2)
library(lattice)
set.seed(123)
confusionMatrix(table(predict_model))

# 5.5 Create the ROC curve and evaluate the AUC value
# install.packages("ROSE")
library(ROSE)
set.seed(123)
roc = roc.curve(mytestdata$new_purchase_test ,  predict_model$predict.best_model..mytestdata..type....response..)
auc = roc$auc
auc

## 6 SVM classification models
#install.packages("e1071")
library(e1071)

# 6.1
# SVM - Polynomial Kernel
# R Code for training data set
set.seed(123)
svm.model1 <- svm(new_purchase_train ~., data=mytraindata, cost=0.01, kernel="polynomial", degree=3, probability=TRUE)
# R Code for Prediction:
set.seed(123)
svm_predict1 = predict(svm.model1,newdata=mytestdata,probability=TRUE)

# SVM - Gausian radial Kernel 
# R code for training data set
svm.model2 = svm(new_purchase_train ~. , data = mytraindata, cost = 0.01, gamma=0.000001, kernel = "radial", probability = TRUE)
# R code for prediction
set.seed(123)
svm_predict2 = predict(svm.model2, newdata=mytestdata, probability = TRUE)

# 6.2 Evaluate the confuction table and calculate Sensitivity, Specificity and Accuracy using the valid data set of prediction
set.seed(123)
svm_df1 = data.frame(mytestdata$new_purchase_test,svm_predict1)
svm_df2 = data.frame(mytestdata$new_purchase_test, svm_predict2)

confusionMatrix(table(svm_df1))
confusionMatrix(table(svm_df2))


# 6.3 Create the ROC curve and evaluate the AUC value
roc_svm1 = roc.curve( mytestdata$new_purchase_test,  svm_df1$svm_predict1 )
auc_svm1 = roc_svm1$auc
auc_svm1

roc_svm2 = roc.curve(mytestdata$new_purchase_test , svm_df2$svm_predict2 )
auc_svm2 = roc_svm2$auc
auc_svm2 
