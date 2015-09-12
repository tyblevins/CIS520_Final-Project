# CIS520_Final-Project
Machine Learning Final Project

## Overview

For this project, you will be developing a system for real estate price estimation: predicting the price of houses given their advertisement.


The dataset is taken from over 40,000 real examples from 7 cities: Boston, Chicago, LA,Miami, NYC, Philly, Vegas. You will be given a training dataset of 20,311 labeled training samples and tested on around 20,307 testing samples. The features of the dataset are binary indicators of the existence of frequent uni-grams and bi-grams in the ads.
Your goal is to predict the (logarithm of) price for the 20,307 test samples.
The format of the project is a competition, with live leaderboards (see below for more details).
Project Rules and Requirements

## Rules and Policies

You CANNOT download or harvest any additional training data from the internet. Both of these will be considered cheating and the penalty will be very harsh. Please, you MUST ONLY use the data we provide you with. We will test your final classifier, and if it is clear that we cannot replicate your performance because you use additional data, you will get a ZERO for the project.
Except when specified otherwise, you are allowed to download additional code or toolboxes from the internet, however, you must cite everything you use in your final project report. We don’t want you to reinvent the wheel. If you’re unsure of what extra resources are allowed, please ask us.


You must work in groups of 2–3 people. No single competitors or groups with more than 3 people will be allowed.
Before you can submit to the leaderboard, you need to register your team using turnin (described below).
In the competition, you need to reach a certain absolute score to get full credit. Placing particularly well will increase your project grade even further. First place gets 10%, second 8%, third 7%, and the rest of the top 10 teams 5% extra credit added to the project grade. The top 3 teams will also get awesome prizes.
You will need to ensure that your code will run from start to finish on our server, so that we can reproduce your result for grading. We will provide a utility so that you can make sure we can run your code successfully. See below for details.
Overall requirements

The project is broken down into a series of checkpoints. There are four mandatory checkpoints (Nov. 20th, Nov. 22nd, Dec. 3rd, and Dec. 6th). The final writeup is due Dec. 11th. The leaderboards will be operating continuously so you can monitor your progress against other teams and towards the score based checkpoints. All mandatory deadlines are midnight. So, the deadline “Nov. 20th” means you can submit anytime before the 20th becomes the 21st.


- 1% - Nov. 20, run turnin -c cis520 -p proj_groups group.txt to let us know your team name. The file should contain a single line: the team name. In order to post scores to the leaderboard, you must have submitted your team name and have 2–3 members of the team total.


- 9% - Nov. 22, Checkpoint: Beat the baseline 1 by any margin.


- 20% - Dec. 3, Checkpoint: Beat the baseline 2 by any margin.


- 50% - By the final submission (Dec 6), implement (or download and adapt an implementation of) at least 4 of the following:
  1. A generative method (NB, HMMs, k-means clustering, GMMs, etc.)
  2. A discriminative method (LR, DTs, SVMs, etc.)
  3. An instance based method (kernel regression, etc.)
  4. Your own kernel (other than the ones such as linear, polynomial, RBF, and sigmoid provided with libsvm)
  5. Your own regularization method (other than the ones such as L_p norm)
  6. A semi-supervised dimensionality reduction of the data


- 20% - Dec 11 Final report: It should be between 2 and 5 pages and should include:
  1. Results for each method you tried (try to use checkpoints to get test set accuracy for each of your methods)
  2. Analysis of your experiments. What worked and what didn’t work? Why not? What did you do to try to fix it? Simply saying “I tried XX and it didn’t work” is not enough.
  3. An interesting visualization of one of your models. For instance, find the words with the most importance for price.


Extra credit - In the competition, placing well will increase your project grade. First place gets 10%, second 8%, third 7%, and the rest of the top 10 teams 5% extra added to the project grade.
Evaluation

## Error metric


Your predictions will be evaluated based on their root mean squared error (RMSE). Your code should produce an Nx1 vector of rating predictions. If each element \hat{y}_{i} is the prediction of the rating of the i^{th} review and y_i is the true label then RMSE is:
\mbox{Root Mean Squared Error} = \sqrt{\frac{1}{N}\sum_{i=1}^N  (y_i - \hat{y}_i)^2}
Requirements for Each Checkpoint


For the second and third checkpoints, you must submit to the leaderboard(s). For the final checkpoint, you must submit ALL of your code via turnin to the correct project folder. Make sure that you submit any code that you used in any way to train and evaluate your method. We will be opening up an autograder that will check the validity of your code to ensure that we’ll be able to evaluate it at the end.
Detailed Instructions

## Download the starter kit

You can download the starter kit here: http://alliance.seas.upenn.edu/~cis520/fall14/project_kit.zip
Inside the code directory, run_submission.m file generates a txt file called submit.txt which you will submit to leaderboard; the other files are the template for your final submission.
Register your team name

Before you can get results on the leaderboard, you need to submit your team name. Everyone on your team is required to do this. Simply create a text file on eniac with your team name as follows:
$ echo "My Team Name" > group.txt
$ turnin -c cis520 -p proj_groups group.txt
This group.txt file should be raw text and contain only a single line. Do not submit PDFs, word documents, rich text, HTML, or anything like that. Just follow the above commands. If you have a SEAS email address, then you will get an email confirmation.
Submit to the leaderboard

To submit to the leaderboard, you should submit the file submit.txt which has 1 number per line for each example in the test set. An example of how to generate this file is in the starter kit.
Once you have your submit.txt, you can submit it with the following:
turnin -c cis520 -p leaderboard submit.txt
Your team can submit once every 5 hours, so use your submissions wisely. Your submission will be checked against the reference solutions and you will get your score back via email. This score will also be posted to the leaderboard so everyone can see how awesome you are.
You can view the current leaderboard here: http://www.seas.upenn.edu/~cis520/fall14/leaderboard.html
Submit your code for the final checkpoint or to test correctness

The file init_model.m will be used to load your model(s) to the workspace. To load your models, you can simply save your models to .mat files and load them within the init_model.m .
The time constraint for initializing your model(s) is 3 minutes.
The file make_final_prediction.m will call the model you loaded and predict test samples one at a time. The inputs of the make_final_prediction.m are the model and test_features which is the concatenation of city, word and bigram. The output is one scalar prediction.
The time constraint for making predictions on 20,307 test samples is 10 minutes.
You must submit your code for the final checkpoint. You can do so with the following:
turnin -c cis520 -p project <list of files including make_final_prediction.m, init_model.m>
You will receive feedback from the autograder, exactly like the homework.
The feedback you will get from the autograder is whether the model is initialized within 3 minutes, whether the final prediction code runs within 10 minutes for 20,307 test samples and whether the submission size is less than 50 Mb. You will not get feedback about your RMSE performance on the test set.
The final rankings will be released on the day of the prize ceremony, Dec. 8.
