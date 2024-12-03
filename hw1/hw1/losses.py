import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # # ====== YOUR CODE: ======
        truth_scores = x_scores[:, y].diag().reshape([x_scores.shape[0],1])
        repeated_truth_scores = truth_scores.repeat(1, x_scores.shape[-1])
        M = self.delta + x_scores - repeated_truth_scores
        maxed_M = torch.maximum(M, torch.zeros(M.shape))
        loss = torch.mean(torch.sum(maxed_M, dim=1) - self.delta)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        # The obvious gradients (Linear)
        self.grad_ctx["C"] = x_scores.shape[1]
        self.grad_ctx["m"] = M
        self.grad_ctx["y"] = y
        self.grad_ctx["x"] = x
        # # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        # Extract params
        m = self.grad_ctx["m"]
        y = self.grad_ctx["y"]
        class_count = self.grad_ctx["C"]
        x = self.grad_ctx["x"]
        # Computing G
        G = (m > 0).float()
        G_row_penalty = -1 * torch.sum(G, dim=1).reshape([m.shape[0], 1])
        G_truth_mask = torch.nn.functional.one_hot(y, num_classes=class_count)
        G_truth_penalty = torch.mul(G_row_penalty, G_truth_mask)
        G += G_truth_penalty
        grad = torch.matmul(x.T, G) / x.shape[0]
        # ========================

        

        return grad
