import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.buffer import Buffer


class LocalDerpp(ContinualModel):
    NAME = 'local-derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--buffer_size', type=int, required=True)
        parser.add_argument('--minibatch_size', type=int, required=True)
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--beta', type=float, default=0.5)
        parser.add_argument('--local_loss_weight', type=float, default=1.0)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        #optimizer 1 &2
        self.opt1 = self.get_optimizer(
            list(self.net.fc1.parameters()) + list(self.net.local_head1.parameters()))
        
        self.opt2 = self.get_optimizer(
            list(self.net.fc2.parameters()) + list(self.net.local_head2.parameters()))
        self.head_optimizers = (self.opt1, self.opt2)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        for optimizer in self.head_optimizers:
            optimizer.zero_grad()

        outputs_all = self.net.forward_all_heads(inputs)

        head_losses = []                #loop over CE for each head
        for head_id, outputs in enumerate(outputs_all):
            weight = (
                1.0 if head_id == len(outputs_all) - 1
                else self.args.local_loss_weight
            )
            head_losses.append(weight * self.loss(outputs, labels))

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits_all = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform
            )
            buf_outputs_all = self.net.forward_all_heads(buf_inputs)

            for head_id, buf_outputs in enumerate(buf_outputs_all):     #loop over heads with old logits
                old_logits = buf_logits_all[:, head_id, :]
                replay_loss = (
                    self.alpha * F.mse_loss(buf_outputs, old_logits) +
                    self.beta * self.loss(buf_outputs, buf_labels)
                )
                weight = (
                    1.0 if head_id == len(buf_outputs_all) - 1
                    else self.local_loss_weight
                )
                head_losses[head_id] += weight * replay_loss

        for head_loss, optimizer in zip(head_losses, self.head_optimizers):    # per head loss
            head_loss.backward()     
            optimizer.step()

        with torch.no_grad():
            logits_to_store = torch.stack(
                self.net.forward_all_heads(not_aug_inputs),
                dim=1
            )

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=logits_to_store.data
        )

        return sum(head_loss.item() for head_loss in head_losses)
