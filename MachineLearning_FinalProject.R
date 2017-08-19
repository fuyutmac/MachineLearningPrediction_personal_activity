library(caret)
library(randomForest)
library(rpart)

## Check file and download
training_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists("pml-training.csv")){
        download.file(training_url, destfile="pml-training.csv",method = "auto")
}

if (!file.exists("pml-testing.csv")){
        download.file(testing_url, destfile="pml-testing.csv",method = "auto")
}



data_training <- read.csv("pml-training.csv",na.strings = c("#DIV/0!", "NA"))
data_testing <- read.csv("pml-testing.csv",na.strings = c("#DIV/0!", "NA"))

dim(data_training)
dim(data_testing)

colnames(data_training)
levels(data_training$classe)



### Cleaning Data
## first 7 columns are distractions, drop them
data_training <- data_training[,-c(1:7)]
data_testing <-  data_testing[,-c(1:7)]
 
## split training data for training and testing the performance
set.seed(08192017)
inTrain <- createDataPartition(y=data_training$classe,p = 0.75, list = F)
training <- data_training[inTrain,]
testing <- data_training[-inTrain,]

dim(training)

## remove the values close to 0
data_near_zero <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,data_near_zero$nzv == F]

## Drop rows which contain to many NAs. Drop NA rate > 0.3 columns
training_clean <- training[lapply(training, function(x) sum(is.na(x))/length(x))< 0.7]
dim(training_clean)

## match testing columns to training columns
col_names <-  colnames(training_clean)
testing_clean <-  testing[, col_names[1:52]]

## Prediction using Random Forest
model_rf <- randomForest(classe~., data = training_clean)
prediction <- predict(model_rf, testing_clean)
cm_rf <-  confusionMatrix(prediction, testing$classe)
cm_rf



## Prediction using Decision Tree
model_rpart <-  rpart(classe~., data = training_clean, method = "class")
prediction_rpart <- predict(model_rpart, testing_clean, type = "class")
cm_rpart <- confusionMatrix(prediction_rpart,testing$classe)
cm_rpart



## Use random forest model to predict
predict_classe <- predict(model_rf, data_testing)
predict_classe

