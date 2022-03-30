
library(remotes)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textmodels)
library(glmnet)
library(randomForest)
library(writexl)
library(caret)
#install.packages("lexicon")

rm(list = ls())

#1. Data preparation  ------------------------------------------------------------
## 1.1 Load data ----

reviews <- read.csv("game_train.csv", encoding = "UTF-8")
games <- read.csv("games.csv", encoding = "UTF-8")
data.kaggle <- read.csv("game_test.csv", encoding = "UTF-8")

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
data$user_review <- gsub("=|░", "", data$user_review) 

data.kaggle$user_review <- gsub("░","", data.kaggle$user_review)


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
  tokens_replace(pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)%>%
  #tokens_wordstem()%>%
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


doc.term <- dfm_trim(doc.term, min_docfreq=20, verbose=TRUE)
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

data.export <- na.omit(data.export)
data.export <- data.export[, -1]
forrest <- rfcv(trainx = data.export, trainy = as.integer(data.train$user_suggestion))



forest <- randomForest(x = data.export, y = data.train$user_suggestion)

#save prediction

data.export <- convert(doc.term, to = "data.frame")
write.csv(data.export, file = "doc.term matrix")
write.csv(prediction.kaggle, file = "predictions.csv", row.names = F)


