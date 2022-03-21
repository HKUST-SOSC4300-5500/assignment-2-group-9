# Assignment 2

### Context: video game reviews

In this assignment we work with data from a video gaming platform. Each game is given a texual review and a yes/no recommendation by users on the platform. We will explore the relationship between the review and the recommendation.

Each row in `game_train.csv` and `game_test.csv` represents a review given to a game. Each game usually has multiple reviews. There are five columns in `game_train.csv`: 

- review_id: the id of the review 
- title: the title of the game reviewed
- year: the year the review was given
- user_review: the review text
- user_suggestion: whether the user recommended the game (1) or not(0)

The `user_suggestion` column is excluded in `game_test.csv`.

`games.csv` include information such as overview and tags on each game included in `game_train.csv` and `game_test.csv`. `example_submission.csv` shows you the format you need to follow in your submission, which is similar to Assignment 1. 

----
### Task

Predict whether a user recommends a game (1) or not (0) using supervised learning. Submit your predictions in this Kaggle competition: 

https://www.kaggle.com/c/2022-hkust-4300-5500-2/

Similar to the grading of Assignment 1, your score will be based on: 

1. (60%) Code notebook (Python notebook or R markdown file) on GitHub:  with clear documentation of methods used and attempts you have tried to improve your prediction performances. We will evaluate based on:
  - **the steps you take to transform the text data for the use of supervised learning**
  - properly tuned model parameters (i.e., not use the default setting but try to find the best tuning parameter)
  - discussions on the strategies you used to improve your model prediction
2. (40%) The best performance you achieved. Here you are competing with your classmates on who can achieve the best performance.

