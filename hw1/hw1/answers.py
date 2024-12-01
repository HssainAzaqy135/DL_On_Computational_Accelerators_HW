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
The test set should only be used at the very end to check how well the model works on completely new data. Given that we use the test set to choose the best params, we are effictively "leaking" information about it into the training process, which makes the final test results unreliable and probably better than it actualy should be in practice.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**



// TODOOOOOOOOOOOOOOO

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
