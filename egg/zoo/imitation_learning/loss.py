import torch
import torch.nn.functional as F


class DiffLoss(torch.nn.Module):
    def __init__(self, n_attributes, n_values, generalization=False):
        super().__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
        self.test_generalization = generalization

    def forward(
        self,
        sender_input,
        _message,
        _receiver_input,
        receiver_output,
        _labels,
        _aux_input,
    ):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(
            batch_size, self.n_attributes, self.n_values
        )

        if self.test_generalization:
            acc, acc_or, loss = 0.0, 0.0, 0.0

            for attr in range(self.n_attributes):
                zero_index = torch.nonzero(sender_input[:, attr, 0]).squeeze()
                masked_size = zero_index.size(0)
                masked_input = torch.index_select(sender_input, 0, zero_index)
                masked_output = torch.index_select(receiver_output, 0, zero_index)

                no_attribute_input = torch.cat(
                    [masked_input[:, :attr, :], masked_input[:, attr + 1 :, :]], dim=1
                )
                no_attribute_output = torch.cat(
                    [masked_output[:, :attr, :], masked_output[:, attr + 1 :, :]], dim=1
                )

                n_attributes = self.n_attributes - 1

                # Tensor: batch_size x n_attributes (a batch of objects, each with n attributes)
                attr_acc = (
                    # average across a batch of objects
                    (
                        # whether the listener got all the attributes right in one vector (about an object)
                        (
                            no_attribute_output.argmax(dim=-1)
                            == no_attribute_input.argmax(dim=-1)
                        ).sum(dim=1)
                        == n_attributes
                    )
                    .float()
                    .mean()
                )
                acc += attr_acc

                #
                attr_acc_or = (
                    # average score on an attribute (averaged across all attributes and objects in the batch)
                    (
                        no_attribute_output.argmax(dim=-1)
                        == no_attribute_input.argmax(dim=-1)
                    )
                    .float()
                    .mean()
                )
                acc_or += attr_acc_or
                labels = no_attribute_input.argmax(dim=-1).view(
                    masked_size * n_attributes
                )
                predictions = no_attribute_output.view(
                    masked_size * n_attributes, self.n_values
                )
                # NB: THIS LOSS IS NOT SUITABLY SHAPED TO BE USED IN REINFORCE TRAINING!
                loss += F.cross_entropy(predictions, labels, reduction="mean")

            acc /= self.n_attributes
            acc_or /= self.n_attributes
        else:
            acc = (
                # whether the listener got all the attributes right in one vector (about an object), batch_size x 1
                torch.sum(
                    (
                        receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
                    ).detach(),
                    dim=1,
                )
                == self.n_attributes
            ).float()

            # whether the listener got each attribute right: batch_size x n_attributes
            acc_or = (
                receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)
            ).float()

            receiver_output = receiver_output.view(
                batch_size * self.n_attributes, self.n_values
            )
            labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
            loss = (
                F.cross_entropy(receiver_output, labels, reduction="none")
                .view(batch_size, self.n_attributes)
                .mean(dim=-1)
            )

        # acc_or: in [0, 1], on average over each attribute, whether the prediction is correct
        # acc: whether the
        return loss, {"acc": acc, "acc_or": acc_or}
