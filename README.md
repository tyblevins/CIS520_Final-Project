# CIS520_Final-Project
Machine Learning Final Project

##Overview

For this project, you will be developing a system for predicting someone’s gender (male/female) from the language of their tweets and the image they post with their twitter profile. You will be given a training dataset of 5,000 labeled training samples and tested on around 5,000 testing samples. The features of the dataset are described in more detail in the slides.
The format of the project is a competition, with live leaderboards (see below for more details).
Project Rules and Requirements

##Rules and Policies

You CANNOT download or harvest any additional training data from the internet. Both of these will be considered cheating and the penalty will be very harsh. Please, you MUST ONLY use the data we provide you with. We will test your final classifier, and if it is clear that we cannot replicate your performance because you use additional data, you will get a ZERO for the project.
Except when specified otherwise, you are allowed to download additional code or toolboxes from the internet, however, you must cite everything you use in your final project report. We don’t want you to reinvent the wheel. If you’re unsure of what extra resources are allowed, please ask us.
You must work in groups of 2–3 people. No single competitors or groups with more than 3 people will be allowed.
Before you can submit to the leaderboard, you need to register your team using turnin (described below).
In the competition, you need to reach a certain absolute score to get full credit. Placing particularly well will increase your project grade even further. First place gets 10%, second 8%, third 7%, and the rest of the top 10 teams 5% extra credit added to the project grade. The top 3 teams will also get awesome prizes.
You will need to ensure that your code will run from start to finish on our server, so that we can reproduce your result for grading. We will provide a utility so that you can make sure we can run your code successfully. See below for details.
Overall requirements

The project is broken down into a series of checkpoints. There are four mandatory checkpoints (Nov. 19th, Nov. 22nd, Dec. 2rd, and Dec. 5th). The final writeup is due Dec. 10th. The leaderboards will be operating continuously so you can monitor your progress against other teams and towards the score based checkpoints. All mandatory deadlines are midnight. So, the deadline “Nov. 19th” means you can submit anytime before the 19th becomes the 20th.

1. 1% - Nov. 19, run turnin -c cis520 -p proj_groups group.txt to let us know your team name. The file should contain a single line: the team name. In order to post scores to the leaderboard, you must have submitted your team name and have 2–3 members of the team total.
2. 9% - Nov. 22, Checkpoint: Beat the baseline 1 (86% accuracy on the test set) by any margin.
3. 20% - Dec. 2, Checkpoint: Beat the baseline 2 by any margin.
4. 50% - By the final submission (Dec 5), implement (or download and adapt an implementation of) at least 4 of the following:
..* A generative method (NB, HMMs, k-means clustering, GMMs, etc.)
..* A discriminative method (logistic regression, decision trees, SVMs, etc.)
..* An instance based method (kernel regression, k-nearest neighbors, etc.)
..* Your own regularization method (other than the standard L0, L1 or L2 penalty)
..* A semi-supervised dimensionality reduction of the data
5. 20% - Dec 10 Final report should be between 2 and 5 pages and should include:
..* Results for each method you tried (try to use checkpoints to get test set accuracy for each of your methods)
..* Analysis of your experiments. What worked and what didn’t work? Why not? What did you do to try to fix it? Simply saying “I tried XX and it didn’t work” is not enough.
..* An interesting visualization of one of your models. For instance, find the words that most correlate with the outcome.

- Extra credit - In the competition, placing well will increase your project grade. First place gets 10%, second 8%, third 7%, and the rest of the top 10 teams 5% extra added to the project grade.

#Evaluation

##Error metric

Your predictions will be evaluated based on their L_0 Err. I.e. the number of predictions you get wrong. Your code should produce an Nx1 vector of predictions, each of which is 0 or 1.
Requirements for Each Checkpoint

For the second and third checkpoints, you must submit to the leaderboard(s). For the final checkpoint, you must submit ALL of your code via turnin to the correct project folder. Make sure that you submit any code that you used in any way to train and evaluate your method. We will be opening up an autograder that will check the validity of your code to ensure that we’ll be able to evaluate it at the end.
