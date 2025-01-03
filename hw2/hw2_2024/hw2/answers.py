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

yes
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


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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