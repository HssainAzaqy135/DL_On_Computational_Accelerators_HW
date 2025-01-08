r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
Q1:

A. The shape comes out to be 64x512x64x1024

B. Yes, the Jacobian is sparse, because it represents how each weight in W affects the predictions. In other words, it shows how each element in the predictions is affected by the weights W. 
For any sample i  such that 1 <= i <=64, we have that the block corresponding to how the sample Xj for some j != i affects the output Yi is completely zero. Thus, the Jacobian is effectively block diagonal with 64 blocks, therefore sparse.

C.No, we do not need to materialize the Jacobian to calculate a downstream gradient with respect to the given input, $\delta\mathbf{X} = \frac{\partial L}{\partial \mathbf{X}}$. This is because the gradient can be computed using the chain rule. Specifically, $\delta\mathbf{X}$ can be obtained as $\delta\mathbf{X} = \delta\mathbf{Y} \mathbf{W}$, where $\delta\mathbf{Y} = \frac{\partial L}{\partial \mathbf{Y}}$ is the gradient of the loss with respect to the output, and $\mathbf{W}$ is the weight matrix. This avoids explicitly constructing the large Jacobian.


Q2:

A. The shape comes out to be 64x512x512x1024

B. No the jacobian is not sparse by any garentee. Since every element of the output Y depends on all elements of W. Given W is any matrix, no sparesity is garenteed.

C. No, we do no need to materialize the Jacobian to compute: $\delta\mathbf{W} = \frac{\partial L}{\partial \mathbf{W}}$. 
We can compute $\delta\mathbf{W}$ directly using the following:

$$
\delta\mathbf{W} = \delta\mathbf{Y}^\top \mathbf{X},
$$

Again avoiding explicitly forming the Jacobian for computation.

"""

part1_q2 = r"""

No, back-propagation is not absolutely required to train neural networks using gradient based optimization , 
yet it makes the training way more feasible.

Back propagation is effectively essential for gradient based optimization:** The fundamental idea behind gradient-based training is to adjust the weights of the neural network in a way that minimizes a loss function. To do this effectively, you need to compute the gradients of the loss function with respect to each of the weights in the network. Doing that the naive way would be way to slow for networks of most medium to large sizes or architectures


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd =  0.1
    lr = 0.1
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.0001
    lr_vanilla = 0.02
    lr_momentum = 0.005
    lr_rmsprop = 0.00025
    reg = 0.0005
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr = 0.0015
    # best informative values 
    # wstd = 0.001
    # lr = 0.0015
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
Q1

A. Comparison of Graphs with Dropout and No-Dropout:

Regarding the graphs with dropout and no-dropout, the results align with our expectations. As anticipated, the highest training accuracy is observed for the no-dropout variant due to its strong overfitting to the training data. This also explains the lower training loss in the no-dropout variant. However, when examining the test accuracy, we observe that the no-dropout variant achieves a lower accuracy compared to the 0.4 dropout variant. This indicates that the 0.4 dropout variant is less overfitted to the training data and exhibits better generalization than the no-dropout variant.



B. Comparison of Graphs with 0.4 Dropout and 0.8 Dropout:

For the graphs comparing 0.4 dropout and 0.8 dropout, we observe that the training loss is lower with 0.4 dropout. This is expected, as fewer neurons and connections in the 0.8 dropout configuration make it harder to effectively minimize the training loss. Training accuracy is also higher with 0.4 dropout, as more active neurons allow for a better fit to the training data. While the test losses are fairly similar, the test accuracy for the 0.4 dropout variant is substantially better. This suggests that the 0.8 dropout variant is underfitted to the data and, as a result, has poor generalization. In contrast, the 0.4 dropout variant demonstrates good generalization and superior performance.

"""

part2_q2 = r"""
Q2


Yes, it is possible for test loss to increase while test accuracy improves during training with the cross-entropy loss function. This happens because loss and accuracy measure different aspects of performance. Cross-entropy loss evaluates both the correctness and confidence of predictions, penalizing less confident predictions for the correct class. In contrast, accuracy measures only the proportion of correct predictions without considering confidence. For example, the model may correctly classify more samples (increasing accuracy) but with lower confidence, leading to higher loss. Regularization techniques like dropout, noisy or hard-to-classify data, or shifts in the model's confidence can also cause this behavior. Such occurrences are often temporary and typically resolve as training progresses, but persistent increases in test loss may require adjustments to the model or training process.

"""

part2_q3 = r"""
Q3

1.

Gradient descent is an optimization algorithm that updates model parameters to minimize a loss function by iteratively moving in the direction opposite the gradient. Backpropagation is a method specific to neural networks that calculates these gradients efficiently using the chain rule. It involves a forward pass to compute the output and loss, followed by a backward pass to determine how each parameter affects the loss. While gradient descent focuses on updating weights, backpropagation focuses on computing the gradients needed for these updates.In neural networks, backpropagation computes gradients, and gradient descent uses them to optimize the model. Gradient descent is general, while backpropagation is tailored to neural networks.

2.

GD and SGD are optimization techniques used to minimize a loss function by updating model parameters iteratively. The main difference lies in how they compute the gradient: GD calculates the gradient using the entire dataset, whereas SGD computes it using a single data point at a time.

GD is computationally expensive for large datasets since it processes the entire dataset in every iteration, making it slower and less practical for massive data but with generally smooth and stable convergence, as it uses precise gradients calculated over all samples. GD is best suited for small or medium-sized datasets and convex optimization problems where accessing the full dataset is feasible.

SGD, on the other hand, is faster per iteration because it processes only one data point at a time, which is randomly selected during each iteration. This random sampling introduces variability in the gradient, helping the algorithm avoid getting stuck in local minima and ensuring that the updates are not biased toward specific subsets of the data. Its convergence is noisier due to high variance in gradient estimates, which can slow progress but also helps escape local minima.

In summary, GD offers precision and stability for smaller datasets, while SGD provides speed and scalability for large-scale machine learning tasks.

3.

SGD is used more often in deep learning due to its scalability, efficiency, and ability to handle large datasets.Traditional GD, requires processing the entire dataset at once. SGD updates parameters using a single data point at a time, making it highly memory-efficient and suitable for massive datasets. This frequent parameter update leads to faster convergence, especially in the early stages of training by is also more noisy. The randomness in SGD helps it escape local minima or saddle points, which is beneficial in the complex high-dimensional menifolds of neural networks. Additionally, SGD can be easily adapted with variants like Mini-batch SGD, Momentum and RMSprop to stabilize learning and improve performance. These characteristics, along with SGD's ability to generalize better and prevent overfitting, make it the preferred optimization method for deep learning tasks.

4.

A.

The friend’s suggested approach would not produce a gradient equivalent to GD. In GD the gradient is computed over the entire dataset, averaging the gradients for each data point. In contrast, the friend’s method involves splitting the data into batches, computing the loss for each batch, summing these losses, and then performing a single backward pass on the sum of the batch losses. This approach leads to a gradient that is not averaged over the entire dataset, but rather computed on the summed batch-wise losses, resulting in a scaled gradient. The key difference is that GD computes:

$\nabla L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla l(\theta; x_i, y_i)$

while the friends approach computes:

$\nabla L_{\text{total}}(\theta) = \sum_{j=1}^{K} \sum_{(x_i, y_i) \in B_j} \nabla l(\theta; x_i, y_i)$

where $N$ is the total number of samples, $K$ is the number of batches, and $B_j$ represents the data in batch $j$. The difference arises because the gradient is summed, not averaged, leading to a larger step size.

However, if we average the gradients across all batches, such that we include all the data and then divide by the total number of samples $N$, then the friend's method would be equivalent to GD.

B.

The out-of-memory error likely occurred because the memory usage accumulated over multiple batches. This can happen if intermediate results, such as activations, gradients, or other data structures, are not properly cleared or released after each batch is processed. This can lead to increased memory consumption  with each batch, eventually exceeding available memory. Additionally, large temporary variables required for forward and backward passes in deep learning models can further contribute to this issue. It can be solved by managing memory efficiently by clearing intermediate result and using memory-saving techniques like gradient accumulation.

"""

part2_q4 = r"""
**Your answer:**
Q4

1.A.

Forward mode automatic differentiation (AD) computes derivatives alongside function evaluations by traversing the computational graph in a forward direction. For a composite function $f$, it calculates both the function values and derivatives at each step. As we have $n$ such function holding these values in memory would take $O(N)$ memory complexity.

To Reduce memory complexity we can discarding Intermediate Values. Instead of storing all intermediate results, only the current and previous values are retained, reducing memory usage to $O(1)$. A trade-off between memory and computation can be achieved by storing only key values (checkpoints) and recomputing others as needed.

B.

We can use a technique called Checkpointing to reduce memory usage in backward mode automatic differentiation (AD) by strategically storing intermediate values during the forward pass and recomputing them as needed during the backward pass. Instead of saving all n intermediate values, the computational graph is divided into $\sqrt(n)$ segments, and only the values at the boundaries of these segments (checkpoints) are stored. This reduces memory complexity from $O(n)$ to $O(\sqrt(n))$.

During the backward pass, any intermediate values required for gradient computations are recomputed by traversing the graph within their respective segments. Since each segment contains $\sqrt(n)$, steps.


2.


Memory reduction techniques such as checkpointing can be generalized to arbitrary computational graphs by strategically storing intermediate values (checkpoints) and recomputing others as needed. In forward mode AD, this involves discarding and recomputing values in a forward traversal, while in backward mode AD, checkpoints are placed at key nodes, and subgraphs are recomputed during the backward pass. They can effective for complex graphs with loops, branches, or parallel paths.


3.

The backpropagation algorithm can significantly benefit from memory reduction techniques, like checkpointing, when applied to deep architectures such as VGGs and ResNets. These architectures have numerous layers, often requiring a large amount of memory to store intermediate activations during the forward pass. Without optimization, this memory demand limits the batch size or the maximum network depth that can be trained on hardware with limited memory.

By applying checkpointing, only selected intermediate activations (checkpoints) are stored during the forward pass, while others are recomputed as needed during the backward pass. This reduces memory usage allowing deeper networks or larger batch sizes to be trained without exceeding memory limits. The trade-off is additional computation for recomputation, which is often manageable compared to the gains in memory efficiency.
"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
//-----------------------------//

A. High optimization error:

High optimization error is when the model *struggles* to minimize the loss function on the training set.
This indicates that the model is unable to effectively fit the data it is trained on (Underfitting).

Possible causes can include:

1. A model with low number of parameters.

2. Choice of optimization algorithm or optimizer parameters.

3. Vanishing/exploding gradients.

4. Too low or too large of a receptive field: The receptive field is the area of input data the model can "observe" or "consider" at a time. A bad receptive field can prevent the model from capturing complex patterns or capture too complex patterns.

Optimizing any of the mentioned above would help with the High optimization error. All are pretty obvious except maybe the vanishing and exploding gradients, in that case we can use gradient norm clamping between to values a and band optimize these parameters.

//-----------------------------//

B. High generalization error:

Correlated with overfitting, a high generalization error occurs when the model performs well on the training set but not that good onthe test set. Effectively the model learned the noise or specific patterns of the training data as is rather than general features of the data set.

Possible causes can include:

1. High model complexity: "too many" parameters can cause overfitting.

2. Training data represents the general dataset badly (Bad sampling, different distributions at training and test times).

3. Lack of regularization and early stopping: supposed to reduce or at least constrain model complexity.

Possible fixes:

1. Lowering model complexity. (Less parameters allowed, for example a narrower or shallower network if using networks).

2. Fix sampling, or using some kind of normalization , maybe even a learnable weighting between distributions at train and test time.

3. Well, add regularizations like L2, L1 or even elastic net.

//-----------------------------//

C. High Approximation Error

High approximation error happens when the model is not expressive enough to capture the underlying complexity of the data. Where the model is too simple to represent the data effectively.

Possible causes can include:

1. Too low model expressive capacity/parameters.

2. Bad feature selection or maybe preprocessing of the input data.

3. Using a non fitting hypothesis class regardless of expressive power (like using a linear model for a non-linear problem).

A helpful principle would be optimizing "Population Loss", Focusing on how effectively the model captures the general data distribution rather than individual samples within said data, using techniques like batch training and ensemble methods.  


"""

part3_q2 = r"""
**Your answer:**

An example for a case with Higher False Positive Rate (FPR) can be spam email detection:

In spam email detection, the system is focused on minimizing missed spam and can becomes overly sensitive, leading to a higher False Positive Rate (FPR). Legitimate emails may be mistakenly flagged as spam and moved to the spam folder. This can cause users to miss important emails. Finding a balance is crucial to avoid excessive false positives while effectively filtering spam.

An example for a case with Higher False Negative Rate (FPR) can be credit card fraud detection:

In credit card fraud detection, a higher False Negative Rate (FNR) occurs when the system fails to identify fraudulent transactions as fraud. This often happens when the model is designed to avoid inconveniencing customers by wrongly flagging legitimate transactions (minimizing false positives). As a result, some actual fraudulent transactions go undetected, allowing fraudsters to continue unauthorized activity. For example, a system might ignore small or unusual purchases that match typical customer behavior, missing signs of fraud.
"""

part3_q3 = r"""
**Your answer:**

The choice of the "optimal" point on the ROC curve depends on the relative costs and risks associated with false positives (FPs) and false negatives (FNs) in each scenario.

In the first scenario, where a person with the disease will eventually develop non-lethal symptoms that confirm the diagnosis, the primary concern is minimizing unnecessary costs and risks from false positives. False negatives are less critical since the disease will eventually be caught and treated. Here, the optimal point on the ROC curve should prioritize minimizing the False Positive Rate (FPR) while maintaining reasonable sensitivity. This approach reduces the number of patients undergoing unnecessary follow-up tests, which are expensive and involve high risks, without compromising patient safety significantly.

In contrast, the second scenario involves a high probability of death if the disease is not detected early. Missing the diagnosis could result in severe consequences, making false negatives critical. In this case, the optimal point on the ROC curve should prioritize minimizing the False Negative Rate (FNR), even if it results in a higher FPR. This involves using a lower classification threshold to ensure that most patients with the disease are flagged for further testing. While this increases the number of false positives, the associated costs and risks are outweighed by the potential to save lives.

"""


part3_q4 = r"""
**Your answer:**


MLPs are not well-suited for sequential data, like text, because they do not capture the order or contextual relationships between elements in a sequence. They treat inputs as independent features, ignoring dependencies such as negations or modifiers critical for understanding sentiment. Additionally, MLPs require fixed-size inputs, which can lead to information loss through truncation or noise from padding. They also lack mechanisms to effectively represent words in context, making them inefficient for natural language tasks. Models like RNNs, LSTMs, or Transformer-based architectures are better alternatives, as they are specifically designed to handle sequential dependencies and retain context for tasks like sentiment classification.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
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
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
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
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""