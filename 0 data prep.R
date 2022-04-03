
library(remotes)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textmodels)
library(glmnet)
library(randomForest)
library(writexl)
library(caret)
library(xgboost)
library(stringr)
library(fastDummies)
library(gmodels)
library(mlr)
#install.packages("lexicon")
#install.packages("gmodels")


rm(list = ls())

#1. Data preparation  ------------------------------------------------------------
## 1.1 Load data ----

reviews <- read.csv("game_train.csv", encoding = "UTF-8")
games <- read.csv("games.csv", encoding = "UTF-8")
data.kaggle <- read.csv("game_test.csv", encoding = "UTF-8")

#see that theres clear differences in reviews!
test.games <- aggregate(data = data,  review_id~title + user_suggestion + is.early, FUN = length)

## 1.2. Combine data ----

data <- merge(reviews, games, by.x = "title", by.y = "title", all.x = T)

data$doc_id <- 1:nrow(data)
train.id <- sample(1:nrow(data), 0.7*nrow(data))
test.id <- data$doc_id[!data$doc_id %in% train.id]

rm(reviews, games)


## 1.3. Clean data ----
### 1.3.1 Remove Tags ----

data$is.early <- ifelse(grepl("Early Access Review", data$user_review), 1, 0)
data$user_review <- gsub("Early Access Review","", data$user_review)
data.kaggle$user_review <- gsub("Early Access Review","", data.kaggle$user_review)

data$received.free <- ifelse(grepl("Product received for free", data$user_review), 1, 0)
data$user_review <- gsub("Product received for free","", data$user_review)
data.kaggle$user_review <- gsub("Product received for free","", data.kaggle$user_review)

data$user_review <- gsub("Access Review", "", data$user_review) #unclear, just remove it
data.kaggle$user_review <- gsub("Access Review","", data.kaggle$user_review)



data$user_review <- gsub("™|¥|â", "♥", data$user_review) 
data$user_review <- gsub("=|▒░", "", data$user_review) 
data$user_review <- gsub("[^[:alnum:][:blank:]?&/\\-\\♥]", "", data$user_review)


data.kaggle$user_review <- gsub("™|¥|â", "♥", data.kaggle$user_review) 
data.kaggle$user_review <- gsub("=|▒░", "", data.kaggle$user_review) 
data.kaggle$user_review <- gsub("[^[:alnum:][:blank:]?&/\\-\\♥]", "", data.kaggle$user_review)



#data$user_review <- gsub("♥", "", data$user_review) 

### 1.3.2 Create Document Term Matrix----

revcorpus <- corpus(x = data, 
                    docid_field ="doc_id", 
                    text_field = "user_review", 
                    meta = list("user_suggestion" = "user_suggestion"))

revcorpus.kaggle <- corpus(x = data.kaggle, 
                           text_field = "user_review")



#stopwords("english")
stopwords.not.included <- c("no", "nor","not","isn't","aren't","wasn't","weren't","hasn't","haven't","hadn't","doesn't", "don't","didn't","won't","wouldn't","shan't","shouldn't","can't","cannot","couldn't","mustn't", "very")

custom.stopwords <- c(stopwords("english")[!stopwords("english") %in% stopwords.not.included])




toks <- tokens(revcorpus, remove_punct = TRUE, remove_number = TRUE, remove_url = TRUE, verbose = FALSE) %>% 
  tokens_remove(pattern = custom.stopwords) %>% 
  tokens_tolower()%>%
  #tokens_replace(pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)%>%
  tokens_wordstem()%>%
  tokens_ngrams(n = 1)
  
doc.term <- dfm(toks)






# doc.term <- dfm(x = revcorpus, 
#                 tolower=TRUE, 
#                 stem=TRUE, 
#                 remove_punct = TRUE, 
#                 remove_url=TRUE, 
#                 verbose=FALSE, 
#                 remove = custom.stopwords)
# 
# doc.term.kaggle <- dfm(x = revcorpus.kaggle, 
#                        tolower=TRUE, 
#                        stem=TRUE, 
#                        remove_punct = TRUE, 
#                        remove_url=TRUE, 
#                        verbose=FALSE, 
#                        remove = custom.stopwords)

topfeatures(doc.term, 20)


doc.term <- dfm_trim(doc.term, min_docfreq=50, verbose=TRUE)
print (dim(doc.term))



#textplot_wordcloud(doc.term, rotation=0, min_size=2, max_size=10, max_words=50)


### 1.3.3 Split into train and test -----

data.train <- dfm_subset(x = doc.term, docname_ %in% train.id)
data.test <- dfm_subset(x = doc.term, docname_ %in% test.id)

### 1.3.4 Match same features -----

data.test <- dfm_match(data.test, features = featnames(data.train))
data.kaggle <- dfm_match(doc.term.kaggle, features = featnames(data.train))

# 2.Train algo -------------------------------------------------------------------

## 2.1. Naive bayes ----------------------------------------------------------------

naive.bayes <- textmodel_nb(data.train, data.train$user_suggestion)
summary(naive.bayes)


##make prediction

actual_class <- data.test$user_suggestion
predicted_class <- predict(naive.bayes, newdata = data.test)
tab_class <- table(actual_class, predicted_class)
tab_class


confusionMatrix(tab_class, mode = "everything")



#make prediction for Kaggle

prediction.kaggle <- data.frame(predict(naive.bayes, newdata = data.kaggle))


prediction.kaggle <- data.frame(matrix(ncol = 2, nrow = nrow(data.kaggle)))
prediction.kaggle[, 1] <- data.kaggle$review_id
prediction.kaggle[, 2] <- data.frame(predict(naive.bayes, newdata = data.kaggle))
names(prediction.kaggle)<- c("review_id", "user_suggestion")


## 2.2 gmlnet  ------------------------------------------------------

lasso <- cv.glmnet(x = data.train,
                   y = as.integer(data.train$user_suggestion == "1"),
                   alpha = 1,
                   nfold = 100,
                   family = "binomial")

prediction.gml <- predict(lasso, data.test, type = "response", s = lasso$lambda.min)

##make prediction

actual_class <- data.test$user_suggestion
predicted_class <- as.integer(predict(lasso, newx = data.test, type = "class", s = lasso$lambda.min))
tab_class <- table(actual_class, predicted_class)
tab_class
confusionMatrix(tab_class, mode = "everything")




#train random Forest -------------------------------------------

data.export <- convert(data.train, to = "data.frame")
data.test.tes <- convert(data.test, to = "data.frame")
data.export <- data.export[, -1]
data.test.tes <- data.test.tes[, -1]


#2.2.3 ---
#try to add game information 
dummies <-  data.frame(data[train.id, "title"])
names(dummies) <- "title"

dummies <- dummy_cols(dummies,select_columns = "title")
dummies$title <- NULL


dummies.test <-  data.frame(data[test.id, "title"])
names(dummies.test) <- "title"

dummies.test <- dummy_cols(dummies.test,select_columns = "title")
dummies.test$title <- NULL

data.export <- cbind(data.export, dummies)
data.test.tes <- cbind(data.test.tes, dummies.test)

dtrain <- xgb.DMatrix(data = as.matrix(data.export),label = data.train$user_suggestion) 
dtest <- xgb.DMatrix(data = as.matrix(data.test.tes),label = data.test$user_suggestion)


params <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eta=0.3, 
               gamma=0, 
               max_depth=5, 
               min_child_weight=1, 
               subsample=1, 
               colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = as.matrix(data.export),label = data.train$user_suggestion,
                 nrounds = 1000, 
                 nfold = 5, 
                 showsd = T, 
                 stratified = T, 
                 print.every.n = 10, 
                 early.stop.round = 20, 
                 maximize = F)

xgb1 <- xgb.train (params = params, 
                  data = dtrain,
                  nrounds = 171, 
                  watchlist = list(val=dtest,train=dtrain), 
                  print.every.n = 10, 
                  early.stop.round = 10, 
                  maximize = F , 
                  eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dtest)
predicted_class <- ifelse (xgbpred > 0.5,1,0)

actual_class <- data.test$user_suggestion
tab_class <- table(actual_class, predicted_class)
tab_class
confusionMatrix(tab_class, mode = "everything")


train <- as.data.frame(data.export)
colnames(train) <- make.names(colnames(train),unique = T)
train$user_suggestion <- data.train$user_suggestion

test <- as.data.frame(data.test.tes)
colnames(test) <- make.names(colnames(test),unique = T)
test$user_suggestion <- data.test$user_suggestion



traintask <- makeClassifTask (data = train ,target = "user_suggestion")
testtask <- makeClassifTask (data = test,target = "user_suggestion")

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", 
                      nrounds=171L)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",
                                          values = c("gbtree","gblinear")), 
                        makeIntegerParam("max_depth",lower = 2L,upper = 6L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
                        makeNumericParam("eta",lower = 0.01,upper = 0.6))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)
ctrl <- makeTuneControlRandom(maxit = 100)

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
mytune$y
mytune$opt.path$par.set$
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)

confusionMatrix(xgpred$data$response,xgpred$data$truth, mode = "everything")





test <- as.matrix(data.export)

xgboost <- xgboost(data = as.matrix(data.export), label = data.train$user_suggestion, 
                   nrounds = 600, #600 , 2, 0.2
                   objective = "binary:logistic",
                   max.depth = 3,
                   eta = 0.08,
                   subsample = 0.6,
                   # min_child_weight = 6,
                   # max_delta_step = 5,
                   gamma = 0.7
                   )


actual_class <- data.test$user_suggestion
predicted_class <- predict(xgboost, newdata = as.matrix(data.test.tes))
tab_class <- table(actual_class, as.numeric(predicted_class > 0.55))
tab_class
confusionMatrix(tab_class, mode = "everything")



predicted_class <- predict(xgboost, newdata = as.matrix(data.export))
actual_class <- data.train$user_suggestion
tab_class <- table(actual_class, as.numeric(predicted_class > 0.5))
tab_class

confusionMatrix(tab_class, mode = "everything")


est_param <- list()
best_seednumber <- 1234
best_error <- Inf
best_error_index <- 0

set.seed(123)
for (iter in 1:100) {
  param <- list(objective = "binary:logistic",
                eval_metric = "error",
                max_depth = sample(1:10, 1),
                eta = runif(1, .01, .1), # Learning rate, default: 0.3
                subsample = runif(1, .8, .9),
                colsample_bytree = runif(1, .5, .6), 
                min_child_weight = sample(1:10, 1),
                max_delta_step = sample(3:8, 1)
  )
  cv.nround <-  300
  cv.nfold <-  5 # 5-fold cross-validation
  seed.number  <-  sample.int(10000, 1) # set seed for the cv
  set.seed(seed.number)
  mdcv <- xgb.cv(data = as.matrix(data.export), label = data.train$user_suggestion, 
                 params = param,  
                 nfold = cv.nfold, 
                 nrounds = cv.nround,
                 verbose = F, 
                 early_stopping_rounds = 8, 
                 maximize = FALSE)
  
  min_error_index  <-  mdcv$best_iteration
  min_error <-  min(mdcv$evaluation_log$train_error_mean)
  
  if (min_error < best_error) {
    best_error <- min_error
    best_error_index <- min_error_index
    best_seednumber <- seed.number
    best_param <- param
  }
  print(iter)
  
}

# The best index (min_rmse_index) is the best "nround" in the model
nround = best_error_index
set.seed(best_seednumber)
xg_mod <- xgboost(data = as.matrix(data.export), label = data.train$user_suggestion,  
                  params = best_param, 
                  nround = nround, 
                  verbose = F)

summary(xg_mod)

# Check error in testing data
predicted_class <- predict(xg_mod, data.test)
actual_class <- data.test$user_suggestion
tab_class <- table(actual_class, as.numeric(predicted_class > 0.5))
tab_class

confusionMatrix(tab_class, mode = "everything")



predicted_class <- predict(xg_mod, newdata = as.matrix(data.export))
actual_class <- data.train$user_suggestion

# 3. try dimensionality reduction ----
#We saw that we could not really get anywhere with our current approach, therefore we try to reduce dimenstionality 
data.export <- convert(data.train, to = "data.frame")
data.export <- data.export[, -1]


reduced_train <- prcomp(as.matrix(data.export), retx = TRUE, center = T, scale. = FALSE, tol = NULL)
reduced_test <- predict(reduced_train, as.matrix(data.test))


reduced_train.mat <- reduced_train$x[, 1:1000]
reduced_test.mat <- reduced_test[, 1:1000]



#data.export <- na.omit(data.export)
set.seed(1)
xgboost <- xgboost(data = reduced_train.mat, label = data.train$user_suggestion, 
                   nrounds = 150,
                   objective = "binary:logistic",
                   max.depth = 2,
                   eta = 0.3, 
)



actual_class <- data.test$user_suggestion
predicted_class <- predict(xgboost, newdata = reduced_test.mat)
tab_class <- table(actual_class, as.numeric(predicted_class > 0.5))
tab_class

confusionMatrix(tab_class, mode = "prec_recall")



predicted_class <- predict(xgboost, newdata = reduced_train.mat)
actual_class <- data.train$user_suggestion
tab_class <- table(actual_class, as.numeric(predicted_class > 0.5))
tab_class

confusionMatrix(tab_class, mode = "everything")



#save prediction----

data.export <- convert(doc.term, to = "data.frame")
write.csv(data.export, file = "doc.term matrix")
write.csv(prediction.kaggle, file = "predictions.csv", row.names = F)


