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
    hypers = dict(
        embed_dim = 128, 
        num_heads = 8,
        num_layers = 3,
        hidden_dim = 100,
        window_size = 16,
        droupout = 0.1,
        lr=0.0002,
    )
    # ========================
    return hypers




part4_q1 = r"""
**Your answer:**

Stacking multiple encoder layers taht have sliding-window attention results in a broader context in the final layer since each additional layer allows information to "persist" beyond the local window. This effect is similar to how stacking convolutional layers in a CNN's effectively increases the receptive field.

How It Works:
In the first Encoder Layer each token attends only to nearby tokens within a fixed size sliding window.No contact between distant tokens yet.
In the second layer and onwards, in a sense, a token can attend to information that was already aggregated from its neighbors within it's window in the previous layer. Given each layer has a sliding window of size 2𝑤 (plus minus w), then after two layers, a token has access to information from about 4𝑤 tokens away (give or take plus or minus 2). then 8w and so on.

"""

part4_q2 = r"""
**Your answer:**

There is the dilated sliding window attention, which instead of attending only to adjacent tokens of the current token. we attend to a dilated set of tokens at each layer. In other words, given a dilation factor of 4, compared to a window of size 8, instead of attending to the four tokens before and the four tokens after, with dilated attention, if the current token is i, we attend to tokens i+1,i+2,i+4 and i+8 and tokens i-1,i-2,i-4, and i-8.
Why would we do this? 
this preserves some of the local context of a token (i+1,i+2,i+4 for example) and with a higher dilation factor this can attend to further away tokens at a decaying density.
Just like a person might read text sometimes and glance a bit further than the current word he is reading and it's neighboring words.
The time complexity stays the same.

"""


part5_q1 = r"""
**Your answer:**
We can see that the fine tuned model performed significantly better that the model that we trained from scratch. The pretrained models are trained on large-scale datasets and exhibit impressive performance, they have good generalization and we can suppose that the google researchers did a good job in building an optimized architecture. As we used a smaller dataset and probably a less advanced architecture, we got subpar performance. However, this will not be the case for every task as the pretrained model was trained to get the optimized results for a specific task or tokens. In this case trining a full model ourselves for this specific task may be better than using a model that is optimized for a different task.

"""

part5_q2 = r"""
**Your answer:**
In this specific case the model is unlikely to finetune succsesfully for the task. The last two layers expect a specific configuration and weights that were trained on a specific architecture for a given task. When we finetune the internal layers we change the capture efficiency of low level features but we do not refine output for the specific task. Therfore the results would be worse.

"""


part5_q3= r"""
**Your answer:**
We cannot use BERT as it is because it lacks a decoder and autoregressive mechanism. The changes that we must add are:

1) add a decoder that can generate target test while keeping the original encoder as source text processor

2) We than have to train the model for the specific task of seq2seq -encoder processes the input text, decoder generates the translated output.

3) Than we should handle input and output tokens in the following way:

The encoder BERT will encode the input into contextual embeddings - 

$embeddings = BERT(X)$

The decoder takes h and generates the target tokens sequentially:

$y_t = Decoder(h,y_1,y_2,...y_t-1)$

Than we should add look-ahead mask in the decoder to prevent the model from seeing future tokens during training and inference.

"""


part5_q4= r"""
**Your answer:**
There are a few reasons to choose RNNs over transformers:

RNN are more data efficient as they learn directly the sequential dependencies in contrast to transformers that need large amount of pretraining, and massive data sets.

RNN are sequential and process one token at a time, while transformers require entire sequences to compute attention. therfore RNN are more useful when fast and light computation is needed for example: in real time speach recognition.

RNN usualy require lower memory usage, as they have lower memory complexity.$O(n) (hidden states for BPTT) in constrast to O(n^2) (Self-attentionn×n matrix)$in transformers in sequence lenght.




"""


part5_q5= r"""
NSP is "Next sequence Prediction". It learns the relation between sequences and its usefull for tasks like answering questions or generating meaningful paragraphs.
The prediction accure in pre training before finetunning the model to specific tasks. 

BERT processes the entire input sequence through its Transformer encoder layers, and in the final layer the  Classification Token (SLT) in extracted. The represantation of the token is than passed to a calssifier to predict which will be the next sentence. 
The binary cross-entropy loss is used between the predicted label and the ground truth.
This loss is combined with the Masked Language Modeling (MLM) loss to optimize the pre-training performance.

It is not a crucial part of pre-training. in RoBERTa, researchers found that removing the NSP loss matches or slightly improves downstream task performance.

ALBERT conjectures that NSP was ineffective because it’s not a difficult task when compared to masked language modeling. In a single task, it mixes both topic prediction and coherence prediction. The topic prediction part is easy to learn because it overlaps with the masked language model loss. Thus, NSP will give higher scores even when it hasn’t learned coherence prediction.

"""





# ==============
