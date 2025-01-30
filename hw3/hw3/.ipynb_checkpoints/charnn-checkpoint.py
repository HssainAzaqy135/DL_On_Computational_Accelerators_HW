import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    chars_to_remove_str = ''.join(chars_to_remove)
    text_clean = text.translate(str.maketrans("", "", chars_to_remove_str))
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    
    result = torch.zeros((len(text), len(char_to_idx)), dtype=torch.int8)
    
    for i, char in enumerate(text):
        idx = char_to_idx[char]
        result[i, idx] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    result = ''
    
    for row in embedded_text:  
        idx = torch.argmax(row).item()  
        char = idx_to_char[idx]  
        result += char  
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    
    # covert to embedding
    embedded_text = chars_to_onehot(text, char_to_idx).to(device)
    
    # splitting into groups of size seq_len
    num_samples = (embedded_text.size(0) - 1) // seq_len #last char has no label
    samples = embedded_text[:num_samples * seq_len].view(num_samples, seq_len, -1)
    
    # make labels
    label_indices = torch.tensor([char_to_idx[char] for char in text[1:]] + [char_to_idx[text[0]]] ).to(device)
    labels = label_indices[:num_samples * seq_len].view(num_samples, seq_len)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    numerator = torch.exp((y-torch.max(y))/temperature)
    # Calculate Softmax
    result = numerator / torch.sum(numerator, axis=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        input_sequence = start_sequence
        hidden_state = None

        while len(out_text) < n_chars:
            # input sequence to one-hot encoded tensor
            input_tensor = chars_to_onehot(input_sequence, char_to_idx).to(dtype=torch.float, device=device)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, seq_len, num_chars)

            output, hidden_state = model(input_tensor, hidden_state)

            # Extract the last chars output, removing the batch dimension
            last_char_output = output.squeeze(0)[-1]

            # Apply temperature-based softmax to get the probabilities for sampling
            probabilities = hot_softmax(last_char_output, temperature=T)

            # Sample a char index
            sampled_char_idx = torch.multinomial(probabilities, num_samples=1).item()
            # Extract char
            sampled_char = idx_to_char[sampled_char_idx]

            # Update the input for the next iteration (use the sampled character)
            input_sequence = sampled_char
            # Append sampled char to generated text
            out_text += sampled_char

    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        idx = []
        
        num_batches = len(self.dataset) // self.batch_size
        
        for start in range(num_batches):
            for i in range(self.batch_size):
                idx.append(start + i*num_batches)
        
        return iter(idx)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: READ THIS SECTION!!

        # ====== YOUR CODE: ======
        dropout_layer = nn.Dropout(dropout)
        
        for k in range(n_layers):
            # Weights initialization
            W_xz = nn.Linear(in_dim, h_dim, bias=False)
            W_xr = nn.Linear(in_dim, h_dim, bias=False)
            W_xg = nn.Linear(in_dim, h_dim, bias=False)
            W_hz = nn.Linear(h_dim, h_dim, bias=True)
            W_hr = nn.Linear(h_dim, h_dim, bias=True)
            W_hg = nn.Linear(h_dim, h_dim, bias=True)

            # Add layers
            self.add_module(f'layer_{k}: W_xz', W_xz)
            self.add_module(f'layer_{k}: W_xr', W_xr)
            self.add_module(f'layer_{k}: W_xg', W_xg)
            self.add_module(f'layer_{k}: W_hz', W_hz)
            self.add_module(f'layer_{k}: W_hr', W_hr)
            self.add_module(f'layer_{k}: W_hg', W_hg)
            self.add_module(f'layer_{k}: dropout', dropout_layer)

            # Add params to layer params list
            self.layer_params.extend([W_xz, W_xr, W_xg, W_hz, W_hr, W_hg, dropout_layer])

            # Update input dimension for next layer
            in_dim = h_dim

        # Final output layer
        self.W_hy = nn.Linear(h_dim, out_dim, bias=True)
        self.add_module('W_hy', self.W_hy)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: READ THIS SECTION!!
        # ====== YOUR CODE: ======
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        output_sequence = []

        for timestep in range(seq_len):
            input_timestep = input[:, timestep, :]
    
            for layer_idx in range(0, len(self.layer_params), 7):
                # Extract the 7 parameters for the current layer
                W_xz, W_xr, W_xg, W_hz, W_hr, W_hg, dropout_layer = tuple(self.layer_params[layer_idx:layer_idx + 7])

                # Get the previous hidden state for the current layer
                hidden_state_prev_layer = layer_states[layer_idx // 7]

                # GRU calculations for update gate (z), reset gate (r), and candidate state (g)
                z = sigmoid(W_xz(input_timestep) + W_hz(hidden_state_prev_layer))
                r = sigmoid(W_xr(input_timestep) + W_hr(hidden_state_prev_layer))
                g = tanh(W_xg(input_timestep) + W_hg(r * hidden_state_prev_layer))

                # Update the hidden state for this layer
                new_hidden_state = z * hidden_state_prev_layer + (1 - z) * g

                # Apply dropout to the new hidden state and update the input for the next layer
                input_timestep = dropout_layer(new_hidden_state)

                # Save the updated hidden state for this layer
                layer_states[layer_idx // 7] = new_hidden_state

            # Append the output of the last layer for the current timestep
            output_sequence.append(self.W_hy(input_timestep))

        # Stack the outputs to form the final sequence output and the hidden states across layers
        layer_output = torch.stack(output_sequence, dim=1)
        hidden_state = torch.stack(layer_states, dim=1)
        # ========================
        return layer_output, hidden_state
