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

Yes, given a cross entropy loss function,it is possible for a test loss to increase while test accuracy improves during training. This loss function evaluates how the predictions are confident in their correctness, giving penalties to less confident decisions. but accuracy measures only the correct predictions without considering confidence. for example the model can correctly classify more samples that results in increased accuracy but with lower confidence, leading to higher loss. There are few things that may cause a situation like this: noisy or complex data and dropout. This often temporary and goes away as we keep training the model.

"""

part2_q3 = r"""
1.

In gradient descent we update model parameters to minimize a loss function by moving in the direction of the steepest descent. In backpropagation we calculate these gradients efficiently using the chain rule. It uses a forward pass to compute the output and loss, and then a backward pass to determine how the loss is affected by each parameter. In short gradient descent updates weights, backpropagation focuses on computing the gradients needed for the weight updates. When discussing neural networks, backpropagation computes gradients, and gradient descent uses them to optimize the model. Gradient descent is a general algorithm, while backpropagation is used in neural networks.

2.

GD and SGD are both optimization techniques that minimize a loss function by updating model parameters. The main difference between them is how they compute the gradient: GD calculates the gradient using the entire dataset, and SGD computes it using a single data point at every time.

GD is expensive computationally  for large datasets because every iteration has to calculate this dataset. Making it slower for massive data but with slow and steady convergence as the calculation are precise because they are performed on the whole database. GD is good for small or medium-sized datasets or convex optimization problems where convergence is fast.

SGD is faster because it processes only one data point at a time which is selected randomly. This random selection helps the algorithm avoid getting stuck in local minimum points and ensure that the updates are not biased toward some small dataset. The convergence will be with bigger noise, which can make progress slower but also can help to escape local minimum point.

In summary, GD offers precise and stable smaller datasets, while SGD provides faster solution that better fits huge datasets.

3.

SGD is used more often in deep learning because it is efficient on large dataset . Traditional GD, requires processing the entire dataset at once. SGD updates parameters using a single data point at each time, making it memory efficient for massive datasets. This parameter makes the convergence faster. The randomness in SGD can help it to escape local minimum points, which is beneficial in the complex high-dimensional menifolds of neural networks. Additionally, SGD can be easily adapted with variants like Mini-batch SGD, Momentum and RMSprop to stabilize learning and improve performance. These characteristics, along with SGD's ability to generalize better and prevent overfitting, make it the preferred optimization method for deep learning.

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

Forward mode automatic diff computes derivatives alongside function evaluations by traversing the computational graph in a forward direction. For a composite function $f$, it calculates both the function values and derivatives at each step. As we have $n$ such function holding these values in memory would take $O(N)$ memory complexity.

To Reduce memory complexity we can discarding Intermediate Values. Instead of storing all intermediate results, only the current and previous values are retained, reducing memory usage to $O(1)$. A tradeoff between memory and computation can be achieved by storing only key values and recomputing others as needed.

B.

we can use a technique called Checkpointing to reduce memory using in backward mode automatic diff by strategically storing intermediate values during the forward pass and recomputing them as needed during the backward pass. Instead of saving all n intermediate values, the computational graph is divided to $\sqrt(n)$ segments, and only the values at the boundaries of these segments are stored. This reduces memory complexity from $O(n)$ to $O(\sqrt(n))$.

Doing the backward pass, any intermediate values required for gradient computations are recomputed by traversing the graph within their respective segments. Since each segment contains $\sqrt(n)$, steps.


2.


memory reduction techniques such as checkpointing can be generalized to arbitrary computational graphs by strategically storing intermediate values and recomputing others as needed. In forward mode AD, this involves discarding and recomputing values in a forward traversal, while in backward mode AD, checkpoints are placed at key nodes, and subgraphs are recomputed during the backward pass. They can effective for complex graphs with loops, branches, or parallel paths.


3.

The backpropagation algorithm can  benefit from memory reduction techniques, like checkpointing, when applied to deep architectures such as VGGs and ResNets. These architectures have numerous layers, often requiring a large amount of memory to store intermediate activations during the forward pass. Without optimization, this memory demand limits the batch size or the maximum network depth that can be trained on hardware with limited memory.

By applying checkpointing, only selected intermediate activations are stored during the forward pass, while others are recomputed as needed during the backward pass. This reduces memory usage allowing deeper networks or larger batch sizes to be trained without exceeding memory limits. The trade off is additional computation for recomputation, which is often manageable compared to the gains in memory efficiency.
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

Possible causes may and can include:

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

1. An example for a case that favours or doesn't care so much about higher false positive rate can be spam email detection:

In spam mail detection, the system is focused on minimizing missed spam and can becomes overly sensitive, leading to a higher fpr.
Normal mails may be falsefully flagged as spams and moved to spam folders and such. Things like this can cause users to miss important mails. Finding a specific balance is important to avoid excessive false positives while effectively filtering said spam at the same time.

2. An example for a case that favours higher false negative rate can be credit card scam detection:

Given the case of credit card fraud detection, a higher false negative rate is when the system fails to identify scams as fraud which we don't want at all.
This often happens when the model is designed to avoid annoying customers by wrongly or falsefuly classifying normal and real transactions trying to minimize false positives.
Therefore, some actual malicious deals go uder the radar, allowing scammers to continue scamming. For instance, a system might ignore small or unusual purchases that match regular clients/customers behaviors, missing signs of scams/frauds since they aren't significant enough.
"""

part3_q3 = r"""
**Your answer:**

The choice of the "optimal" point on a ROC curve depends on the costs and risks correlated with false positives and false negatives in each scenario.

In the first scenario, where a person with the disease will eventually develop non-lethal symptoms that confirm the diagnosis, the primary concern is minimizing unnecessary costs and risks from false positives. False negatives are less crucial since the disease will eventually be identified and treated. In this case, the optimal point on the ROC curve should prioritize minimal FPR while maintaining balanced sensitivity.

On the other hand, the second scenario includes a high probability of death if the disease is not found on the patient as early as possible. Miss classifying could be very bad, making false negatives is critical.
In such a case, the best point on a ROC curve should prioritize minimizing the false negative rate, even if it results in a higher FPR. This can involve using a lower threshold to make sure that most patients with the disease are classified correct. Still this will increase the number of false positives, the associated costs and risks are outweighed by the potential saved lives.

"""


part3_q4 = r"""
**Your answer:**

MLPs are not well suited for sequential data, like text, because they do not capture the order or contextual relationships between elements in a sequence. They treat inputs as independent features, ignoring dependencies such as negations that are important for understanding sentiment. 
In addition, mlps require inputs of fixed sizes and dimensions, which can cause information loss through truncating the inputs or noise from padding said inputs. 
They also lack the ability to effectively represent words in a certain context, making them not so good for natural language tasks. Models similar to RNNs, LSTMs are better options. We saw that they are better for tasks like sentiment analyse.

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
    loss_fn = torch.nn.CrossEntropyLoss();
    lr = 0.01
    weight_decay = 1e-4
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
Q1.1.

Calculation of the number of parameters of regular block:

The regular block has two 3X3 convolutional layers directly operating on a 256-channel input therfore the number of paramas in first layer is:

3 X 3 X 256 X 256 + 256 (bias) = 590080

Second layer is exactly the same, So the final result of the total number of parameters is:

2 X 590080 = 1180160 parameters

Calculation of the number of parameters of bottleneck block:

First layer 1x1 conv 256 to 64:

1 X 1 X 256 X 64 + 64 (bias) = 16,448

Second layer 3x3 conv 64 to 64:

3 X 3 X 64 X 64 + 64 (bias) = 36,928

Final layer 1x1 conv 64 to 256:

1 X 1 X 64 X 256 + 256 (bias) = 16,640

Total number of parameters: 16,448 + 36,928 + 16,640 = 70,016

2.
Number of floating point operations can be approximately calculated in the following way:
For the regular block:

2(convolution layer) X H X W (spatial input size) X 3 X 3 (conv kernal size) X 256 (input channels) X 256 (output channels) = H X W X 1,179,648

For the bottleneck block:

First layer: H X W (spatial input size) X 1 X 1 (conv kernal size) X 256 (input channels) X 64 (output channels)

Second layer: H X W (spatial input size) X 3 X 3 (conv kernal size) X 64 (input channels) X 64 (output channels)

Final layer : H X W (spatial input size) X 1 X 1 (conv kernal size) X 64 (input channels) X 256 (output channels)

Total Flops : H X W X 69,632

3.

The regular block and the bottleneck block differ in their ability to combine input spatially (within feature maps) and across feature maps. The regular block uses two 3×3 convolutions, both operating on the full number of input channels (256). This allows it to effectively combine spatial information within feature maps over a larger receptive field (equivalent to 5×5 for the two stacked layers). As it retains the full number of channels throughout, the regular block is better suited for capturing fine spatial details and dependencies.

On the other hand, the bottleneck block includes one 3×3 convolution for spatial combination, but it operates on a reduced number of feature maps (64, reduced from 256 by a preceding 1×1 convolution). While the effective receptive field is still 5×5, the bottleneck block is less rich in spatial combination due to the dimensionality reduction. However, this reduction also makes the bottleneck block computationally more efficient.

Where the bottleneck block excels is in combining information across feature maps. The 1×1 convolutions in the bottleneck block allow for selective compression and expansion of features, enabling more flexibility in feature maps and gives less weight to redundant ones. In contrast, the regular block lacks this explicit feature selection and recombination mechanism, as both its convolutions are spatially focused.

In summary, the regular block is better at spatially combining information within feature maps, while the bottleneck block is more effective at combining information across feature maps as a result of 1×1 convolutions. 
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

Part1: 

We clearly see that we seam to reach peak performance at **2** layers, and we think this is because of mainly two things. First, we reduced the dataset to around 128*250 = 320000 training examples and and about 5400 test samples, so maybe in a sense 4 layers or more is way too much  parameters to train like we saw in class given this experiment was done using a basic cnn.
Moreover, we first tried using all 60k samples for training and 10k for testing and actually L = 4 came out on top. We reduced the dataset for runtime purposes. So we conclude that with enough samples, more layers should be better given unbound or very large compute power and time. 

Part2:
We clearly see that for L=8,16 the network wan't trainable . This may be due maybe vanishing gradients, or the fact that the deeper the network the more pooling we do with this setup. Maybe the "information loss" is way too much together with vanishing gradients for a network to be trainable.

"""

part5_q2 = r"""
**Your answer:**
First of all, compared to experiment 1_1, in this experiment we get a significant bump in training accuracy and a mild bump in test accuracy across the board.
Interestingly, in this case we see that more kernels are better given more layers. which should mke sense since the kernels maybe are not "utilized" enough with low layers like 2 or 3, yet with 4 they help way more.
All networks here are trainable , since they are not that deep and the kernels theselves aren't effected by the problems stated before for the same reason.

"""

part5_q3 = r"""
**Your answer:**

We can see clearly here that 2 layers perform way better than 3,4. This maybe due to various factors, we think that maybe this is caused by the first layer extracting some "mapping" and the rest of the layers are just added noise and trainable parameters that are dependant at training time on the many kernels applied beforehand.
We have this suspision since for L = 3,4 the accuracy for train and test time is 10%, which isn't good, not that much better than guessing 1 out of 10 classes.

"""

part5_q4 = r"""
**Your answer:**

This time we are using a ResNet architecture.
As shown in the plots, in the first part, there is no vanishing gradients problem. we have that with k = 32 , the more layers the better. There is no vanishing gradients problem. Notice that even though the performance on the test set stayed the same compare to 1_1 and 1_3 the same. the training accuracy is closer to the test accuracy, suggesting way less overfitting.

In the second part , with way higher K, we have that for L = 4 the performance peaked. Specifically the train and test accuracy are way better than 1_1 and 1_3.
We conclude that there is a specific balance to be maintained between K and L. Not a linear relation.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
Q1

1. 

The model performance was poor. The model did misclassifications such as identifying dolphins as people and failed to differentiate between dogs and cats. Also the confidence was relatively low in most of the classifications.

2.

Firstly the yolo5s is trained on the COCO dataset, which has predefined classe.
COCO does not include a specific class for dolphins, so the model tries to match dolphins to the closest available class, often person due to the similarity in shape. also yolov5s is a lightweight version designed for speed, which sacrifices some accuracy. Differentiating between dogs and cats can be challenging for models trained on COCO, especially in unusual contexts, poses, or lighting.

If we are not constrained by hardware or inference time, we can use larger YOLOv5 models. These models are more accurate but require more computational resources. We can also clone the model and train it ourselves with a specialized labeled dataset that includes classes like dolphins, dogs, and cats in various positions and lighting conditions.

3.

To attack an object detection model like YOLO using PGD, the goal is to generate small perturbations in the input image that mislead the model's predictions. YOLO relies on both classification and bounding box predictions. In PGD, the process starts by taking a copy of the original image, then iteratively adjusting it using the gradient of the loss function, which increases the model's error. The perturbation is constrained to avoid noticeable changes to the image while causing incorrect predictions.

The attack targets both the classification (misclassifying objects) and localization (distorting bounding boxes or missing objects). After several iterations, the perturbations are refined to maximize the model's failure in detecting or correctly classifying objects. The effectiveness of the attack is evaluated by checking whether the model misclassifies, fails to detect, or inaccurately locates objects in the image.


"""


part6_q2 = r"""
**Your answer:**

"""


part6_q3 = r"""
**Your answer:**


The model did terrible.
Most humans can recognize that these pictures are of cats.(maybe except the third image).
The model has mistaken the cat in the first pictured to a sheep because of fur and the white carpet.
It misclassified the cat to a dog, because it really resembles a dog but you can clearly see cat features.
Somehow it misclassified the cat in the third picture to a toilet... probably because of the blur.
"""

part6_bonus = r"""
**Your answer:**

1.

By removing the background and adding distinct features, like cat eyes, we effectively made the cat more recognizable. The absence of a distracting or camouflaging background allowed the model to focus solely on the key features of the cat, improving recognition accuracy.

2.

We changed the cat nose and ears to make it resemble a cat better.


3.

we added a cat head and removed the blur a little bit.


This changes made the model recognize the cat correctly.

"""