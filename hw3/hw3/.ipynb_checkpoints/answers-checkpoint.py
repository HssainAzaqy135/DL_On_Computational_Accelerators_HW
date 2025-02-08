r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 80 # all chars, start and end
    hypers['h_dim'] = 100
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 2e-3
    hypers['lr_sched_factor'] = 1e-1
    hypers['lr_sched_patience'] = 4
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = """TRAM. What would you have?
  HELENA. Something; and scarce so much; nothing, indeed.
    I would not tell you what I would, my lord.
    Faith, yes:
    Strangers and foes do sunder and not kiss.
  BERTRAM. I pray you, stay not, but in haste to horse."""
    temperature = 0.3
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

    We do the split for two reasons.
    First of all, if trained on the entire corpus at once, the model will most certainly overfit. Since this is a single data point.
    Moreover, since all the model trained for was memorizing a huge corpus, it will probably have some serious generalization issues.
    By splitting the corpus to many smaller sequences, we get much more variety while we train the model.
"""

part1_q2 = r"""
**Your answer:**

    The hidden state inside the model doesn't depend on the input sequence length. Therefore when generating text, the invoked text can be of different length than the
    input text.
"""

part1_q3 = r"""
**Your answer:**

    while training the network we pass on the hidden state between batches. This state serves as "context" from the previous samples in the batch. This forces us to send
    batches that contain sentences from the same context, hoping that text that is locally close is of the same context generally.
    We do this to train the model for using the hidden state as previous context and not some random value.
"""

part1_q4 = r"""
**Your answer:**
1. The temperature controls how uniform or "spiky" the diffusion of the distribution is. The hotter the more uniform, the colder the more "spiky".Since we aim to use the model to pick the most likely character to come next, we would want the "spiky" distribution, so we use lower temperature which allows
shifting the probability of choosing the right character to be higher and lower other probabilities.
    
2. When the temperature is very high, the distribution becomes uniform, since from the temperature softmax formula we divide all exponents by T. Exponents of numbers close to zero are 1, making all exponents pretty much the same. This is what results in uniform behavior.
    
3. When dividing the exponent inputs by low T, we are making them larger. This, together with the fact they are inputs of exponents, allows us to exaggerate minor differences even more. Resulting in a more "spiky" distribution which allows us to make "stronger" / "more confident" choices with the outputted probabilities.
    

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 10
    hypers["h_dim"] = 1024
    hypers["z_dim"] = 20
    hypers["x_sigma2"] = 0.005
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.9,0.99)#(a,b) place holder
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The sigma2 hyperparameter controls how much the reconstructed output will deviate from the input. it also controls the similarity of the latent space probability distribution to the Normal distribution with mean of 0 and variance of 1.

For low value of sigma2:

The reconstruction error gets a strong penalty - the model will match the input closely.
The model overfits and may ignore KL divergence as it will focus specifically to optimize the reconstruction term

For high value of sigma2:

The reconstruction loss gets scaled down - matching the input precisely is now less important.
optimazing KL divergence is more important. We will get blurry reconstructions as the latent space will be more generalized at the cost of low quality reconstructions.


"""

part2_q2 = r"""
**Your answer:**

---------- Q2.1 ----------

The purpose of both parts of the VAE loss term - reconstruction loss and KL divergence loss:

Reconstruction loss - defined as the mse between the input image and recomstructed image and scaled by sigma 2. The purpose is to define a loss function that measures the resembelncy between the output image and the input image scaled by a factor of "randomness" that gives us room for getting unique images

KL divergence loss - its purpuse is to match the learned latent space to a normal distribution, making it smooth and continues. This produces a combination of the original images with the latent space and prevents overfitting by balancing between generelezation and exact reconstruction.

---------- Q2.2 ----------

The KL loss term shapes the latent space distribution by making it close to a standard normal distribution, in other words normal with mean 0 and standard deviation 1.

---------- Q2.3 ----------

As we said above:

It makes the results smoother and more continious, It prevents overfitting to the learned inputs which makes the model generalize better. 

"""

part2_q3 = r"""
**Your answer:**

Generally, we don't know what is the evidence distribution p(X), it is extremely complex and  not computationally feasible. Yet, our goal is to learn an estimate of p(X) in order to generate other data points.
We do so by maximizing a lower bound for log(p(X)) and we aim to get it as tight as possible so that we get an estimation of p(X) that is accurate using said bound.


"""

part2_q4 = r"""
**Your answer:**

We model the log-variance of the latent space rather than modeling the variance itself for teh following reasons:

1. numerical stability: variance must be strictly positive, but  general neural networks can output any number. If we directly model the variance itself, ensuring positive values requires extra constraints , which can be numerically unstable due to scale or computation method. 
Instead, we let the network predict log-variance, which can take any real value from minus infinity to infinity, and then compute the variance by taking an exponent. This ensures positivity due to exponents being positive.

2. more stable KL divergence computation: in a VAE we use the KL divergence which contains a term log(var^2) which is directly available if we model the log-variance, simplifying calculations and preventing numerical instabilities due to the approximation of logarithmic functions.


"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=8,
        z_dim=100,
        data_label=1,
        label_noise = 0.2,
        learn_rate=0.0002,# not used really
        discriminator_optimizer=dict(
            type="Adam",
            lr=0.0002,

        ),
        generator_optimizer=dict(
            type="Adam",
            lr=0.0002,
        )
    )
    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**

Training the GAN model is done by optimizing the losses of the discriminator and generator, seperately for each batch.

Discriminator training:
First sample a datapoint using the generator and check what is the discriminator's classification and update the discriminator based on the its loss. In this part we do not maintain the gradients we pass to the "sample" function with_grad=False since we don't need to update the generator's weights because of this sampling.  

The generator is trained by generating a datapoint and then showing it to the discriminator and updating the generator based on the its loss.this time we do maintain the gradients (we pass to the "sample" function with_grad=True) since we do want the generator's weights to be updated during backpropagation. 

"""

part3_q2 = r"""
**Your answer:**

1. No, we shouldn't decide to stop training solely based on the fact the generator loss is below some threshold. Our training goal for the generator is that the discriminator becomes as good as flipping a coin when trying to decide if the inputs from the generator are real or not.
This goal lets the generator aim towards beating an "evolving" dicriminator with time.

2. If the discriminator loss remains at a constant value while the generator loss decreases, this may imply that the generator started overfitting the datapoints. Or even worse, the entire model is failing to converge. Since this means the generator is able to fool the discriminator more ofter as we train but the discriminator is not getting better at identifying generated data. And since both the discriminator's and generator's training depend on one another, this means the discriminator is not improving and is just stuck.
We saw this kind of behaviour with Moshe in the tutorial.


"""

part3_q3 = r"""
**Your answer:**

Given the VAE and GAN results we can see that the VAE results are blurry with not as sharp edges while the GAN results are more detailed.
We think this is the expected result since the VAE loss has a reconstruction loss, forcing the output to be close to the input in terms of mean squered error loss which results in smooth edges.
On the other hand in GAN, the generator does not have access to real images while training but learns how those should look through what the discriminator decides. This makes sure the generators predictions to be more realistic since otherwise the discriminator would have an easy time detecting generated images.

"""



PART3_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part4_q1 = r"""
**Your answer:**

"""

part4_q2 = r"""
**Your answer:**


"""


part5_q1 = r"""
**Your answer:**


"""

part5_q2 = r"""
**Your answer:**


"""


part5_q3= r"""
**Your answer:**


"""

# ==============
