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
    hypers['dropout'] = 0.25
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
    start_seq = "to be or not"
    temperature = 5e-4
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
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.9,0.999)#(a,b) place holder
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**

"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = None


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




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
