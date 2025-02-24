import torch
from torch import nn
from torch.utils import checkpoint


################ NEURAL NETWORK ARCHITECTURE
class MLP(nn.Module):
    def __init__(self, use_checkpoint=False, checkpoint_fn=None):
        super().__init__()

        # whether to apply activation checkpointing
        self.use_checkpoint = use_checkpoint
        self.checkpoint_fn = checkpoint_fn

        self.block_1 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.Linear(512, 700),
        )
        self.block_2 = nn.Sequential(
            nn.Linear(700, 512),
            nn.ReLU(),
            nn.Linear(512, 700),
        )
        self.block_3 = nn.Sequential(
            nn.Linear(700, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        if self.use_checkpoint:
            if self.checkpoint_fn is not None:
                out = self.checkpoint_fn.apply(self._checkpoint_layers, x)
            else:
                out = checkpoint.checkpoint(self._checkpoint_layers, x)
        else:
            out = self._checkpoint_layers(x)

        out = self.block_3(out)
        return out

    def _checkpoint_layers(self, x):
        return self.block_2(self.block_1(x))


################ MEMORY CONSUMPTION UTILITY
def get_mem_consumption(model, device):
    torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(1024*3, 3000, device=device, requires_grad=True)
    out = model(x)
    max_mem = torch.cuda.max_memory_allocated(device) / 1e+6
    out.backward(torch.ones_like(out))

    return max_mem


################ PYTORCH HOOKS
def register_forward_hooks(model):

    # function called during the forward pass
    def forward_hook(module, inp, out):
        # for layers where `requires_grad=False`, the activations are 
        # re-computed during the backward pass
        print(f"Forward pass for `{module.__class__.__name__}`: Activations stored = {out.requires_grad}")

    # register hooks to each layer in the model
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Sequential):
            for i in range(len(layer)):
                layer[i].register_forward_hook(forward_hook)


################ CUSTOM CHECKPOINTING FUNCTION
class CustomCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, inputs):
        # save the inputs & function transformation for the layer
        ctx.save_for_backward(inputs)
        ctx.func = func

        # return the output of the layer
        with torch.no_grad():
            return func(inputs)

    @staticmethod
    def backward(ctx, grad_outputs):
        # get the inputs saved from forward pass
        inputs, = ctx.saved_tensors
        # re-compute the activation/output
        with torch.enable_grad():
            outputs = ctx.func(inputs)

        # compute the gradients for the layer
        return (None, *torch.autograd.grad(outputs, inputs, grad_outputs))


################ BENCHMARKING FUNCTION
def run_benchmark(use_checkpoint, checkpoint_fn=None, verify=False):
    device = "cuda"

    # get the model
    model = MLP(use_checkpoint, checkpoint_fn).to(device)
    if verify:
        register_forward_hooks(model)
    # forward & backward pass, return memory consumption
    mem_consumption = get_mem_consumption(model, device)

    print(f"Memory consumption with `use_checkpoint={use_checkpoint}`: {mem_consumption:.2f} MB")


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")

    # our implementation of activation checkpointing
    # change the value to `None` to use PyTorch's default checkpointing
    checkpoint_fn = CustomCheckpointFunction

    # benchmark with no activation checkpoint
    run_benchmark(use_checkpoint=False, verify=True)

    print(f"\n{'-'*60}\n")

    # benchmark with activation checkpoint
    run_benchmark(use_checkpoint=True, checkpoint_fn=checkpoint_fn, verify=True)
    