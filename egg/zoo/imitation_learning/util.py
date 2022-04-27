import torch

def get_grad_norm(agent):
    with torch.no_grad():
        total_norm = 0
        for p in agent.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

def get_avg_grad_norm(agent):
    with torch.no_grad():
        grad_norms = [p.grad.data.norm(2).item() for p in agent.parameters()]
        print(grad_norms)
        return sum(grad_norms) / len(grad_norms)

def get_max_grad(agent):
    with torch.no_grad():
        return max([math.abs(p.grad.data.item()) for p in agent.parameters()])

def sum_grads(agent):
    with torch.no_grad():
        grad_norms = [p.grad.data.norm(2).item() for p in agent.parameters()]
        print(grad_norms)
        return sum(grad_norms)