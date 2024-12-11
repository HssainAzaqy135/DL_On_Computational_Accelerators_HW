r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

We claim that the first two statements are false while the other two are true.

Statement 1. false:

* The in-sample error refers to the training error, which evaluates how well the model fits the training set it was trained on.
The test set is unrelated to the training error because the test set and training set are disjoint. 
The test set, which the model does not see during training, is used to compute the test error (or out-of-sample error), which determines how well the model generalizes to unseen data.

Statement 2. false: 

* We claim that not all train/test splits are equally useful. A "good" split makes sure that the train and test sets accurately represent the underlying data distribution in the sample set. Take for example, if the training set contains samples from only one class or label where as the dataset contains multiple classes, said split won’t represent the overall data distribution, leading to poor model performance on unseen classes compared to the seen class in the training period.

Statement 3. true: 

* CV is conducted exclusively on the training set to avoid leakage of the unseen test set.
In the CV process, the training set is divided into k folds for some integer k. Each fold is used as a validation set once, while the remaining folds are used for training.
This ensures the model is evaluated on unseen data within the training set only. The test set is reserved for evaluating the final model's performance on completely unseen data. (We use CV for tuning hyperparameters in a more accurate manner than using one fold)

Statement 4. true: 

* As explained earlier, in CV, the model does not have the validation fold while being trained on the other folds.
This ensures that the validation fold can be used to estimate the model's performance on unseen data which is the generalization error by imitating a test set.
By averaging the performance across all validation folds, we get a more reliable estimation of the model's ability to generalize.

"""

part1_q2 = r"""
**Your answer:**

The friend's approach isn’t correct.

When tuning a model’s hyperparameters, we should only check how it performs on the training set, by using CV to find the best params.
The test set should only be used at the very end to check how well the model works on completely new data. Given that we use the test set to choose the best params, we are effectively "leaking" information about it into the training process as we incorporate the test set into the model development process. The test set should be completely independent and unseen data set for the final evaluation. The friend's approach also risks overfitting to the test set, as the $\lambda$ chosen may perform well on the test set itself but it might be worse on new unseen data. Therefore, the correct approach would be only using the training set and CV to fine tune $\lambda$ and evaluate exactly once on the test set to measure final performance.


"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
The $\Delta$ in the structured hinge loss represents a penalty that determines the amount of separation enforced between the true class score to the incorrect one. Therefore when $\Delta$ is positive there is a penalty when the model prediction is too close to the boundary. That means that a positive delta enforces stricter separation. It improves robustness as the model becomes less likely to misclassify borderline cases. However, the regularization term encourages smaller weight to prevent overfitting and making it harder for the model to achieve strict separation. In effect regularization encourages the model to generalize better but at the cost of slightly relaxing the separation margin.
In the other case, when $\Delta$ is negative we allow the score of another class to be lower than the true class. It thereby relaxes the requirements for the correct class to have higher score. That decreases robustness, makes weaker generalization and paving the road for numerous misclassifications.



"""

part2_q2 = r"""
**Your answer:**
By looking at the weights it may seem hard to distinguish between the numbers and pinpoint exactly what weights correspond to each number, but we can try to outline some features that the model is classifying upon. for example, by looking at the color of the weighted picture we can get some insights: bright areas (positive weights) are where the model looks for evidence of the digit and gives higher score for pixels that appear in them. In contrast to dark areas where pixels that appear in them will be penalized. Therefore, the model is learning the weights that generalize the weighted map of pixel importance to represent a digit. For example: we can see that in the digit zero the model expects a dark oval circle in the center of the frame, a feature that is common for all zeros. That means that the model will penalize greatly white pixels that appear in the middle. An example of possible classification error is mistaking 4 for a 9 or vice versa. As variation in the form of the digits may correspond to higher score in the weighted map of the other digit that will result in a misclassification. That is a direct result from the relative geometric similarity.

"""

part2_q3 = r"""
**Your answer:**

based on the training set loss graph we can see that The training loss decreases steadily and smoothly over epochs, which indicate a good training rate.
based on the accuracy graph we can see that in the last ephochs Training accuracy is slightly higher than test accuracy also the test accuracy plateaus but remains relatively close to the training accuracy, therfore the model is slightly overfitted to the Training Set


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The ideal pattern to see in a residual plot should be is 0 error therfore we would want a single straight line on the zero mark.Ideally we will get mse of 0.

The fitness of the trained model seems to be fine. As we can see that the training points are clustered closely and similarly to the test points.

The residual plot after cross-validation indicates that the trained model performs better than the model with the top five features. The final plot shows a more consistent residual pattern, with residuals closer to and more centered around the zero line. This is further supported by the lower mean squared error (MSE). Additionally, there are noticeably fewer outliers in the final model's residuals.
"""

part3_q2 = r"""

**Your answer:**

It is still a linear regression model because it is still a linear combination of the features - the model is still a linear function but with just non linear feature mapping. 

Technically we can apply any non-linear on the original features, but it will benefit the model if it captures the underlying non-linear relationships. Meaning we favore features that help seperate the data. However adding features increases the dimension which can lead to overfitting "The curse of dimensionality" .

The decision boundary remains a hyperplane in the transformed feature space because the model is a linear combination of the transformed features. However, in the original feature space, the decision boundary appears non-linear. For instance, if we start with two features and transform them into a higher-dimensional feature space, the decision boundary remains a hyperplane in that transformed space. In contrast, in the original feature space, this boundary may take on a curved shape with non-linear form.
"""

part3_q3 = r"""
**Your answer:**

Using a logarithmic scale enables sampling from a wider range of values while maintaining proportions to the scale. This approach ensures that smaller and larger values are represented relative to their magnitude. In contrast, linear spacing is inefficient for sampling over a large range because it spaces values evenly, often resulting in many samples clustered closely together, leading to redundancy.This reduces the number of samples by narrowing the range. which is good for CV since it has less combinations this way and eases computations.




The model underwent 180 fits to the data. Here's the calculation:
The degree parameter has 3 possible values: [1, 2, 3].
The other parameter includes 20 values.
For each combination the model fitted 3 folds.


"""

# ==============

# ==============
