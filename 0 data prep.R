#1. load data and fix encodings ------------------------------------------------

reviews <- read.csv("game_train.csv", encoding = "UTF-8")
games <- read.csv("games.csv", encoding = "UTF-8")

#2. Combine data----------------------------------------------------------------

data <- merge(reviews, games, by.x = "title", by.y = "title", all.x = T)
rm(reviews, games)

#3. Clean data

data$is.early <- ifelse(grepl("Early Access Review", data$user_review), 1, 0)
data$user_review <- gsub("Early Access Review","", data$user_review)

data$received.free <- ifelse(grepl("Product received for free", data$user_review), 1, 0)
data$user_review <- gsub("Product received for free","", data$user_review)

data$user_review <- gsub("Access Review", "", data$user_review)

#
data$user_review <- tolower(data$user_review)

data$user_review <- gsub("[#&~%:*.!?,$\"/\\-]", "", data$user_review)
data$user_review <- gsub("[0-9]", "", data$user_review)


datacorpus <- corpus(data$user_review)
datacorpus
summary(datacorpus)
