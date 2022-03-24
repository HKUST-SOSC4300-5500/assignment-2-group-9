
data <- read.csv("game_train.csv")
games <- read.csv("games.csv")


test <- merge(data, games, by.x = "title", by.y = "title", all.x = T)
