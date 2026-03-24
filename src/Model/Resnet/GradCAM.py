import torch
import torch.nn as nn

"""
Note that this implementation was heavily guided by claude using it as a mentor to help me understand the concept of 
Grad-CAM and how to implement it in PyTorch. The code is based on the original Grad-CAM paper by Selvaraju et al. (2017) 
and adapted for use with PyTorch models.
"""

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, args, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_full_backward_hook(backward_hook))

    def generate(self, image):
        # Get Prediction
        output = self.model(image)
        pred_class = output.argmax(dim=1)
        cam = None

        if pred_class == 1:
            # Back Prop through prediction weights
            self.model.zero_grad()
            output[0, pred_class].backward()

            # Global Average Pooling
            weights = self.gradients.mean(dim=[2, 3])

            # Weighted combination
            cam = (weights[:,:, None, None] * self.activations).sum(dim=1)

            # ReLU and normalize
            cam = torch.relu(cam)
            cam = cam / cam.max()

            # Resize
            cam = torch.nn.functional.interpolate(cam.unsqueeze(0), size=(256, 256), mode='bilinear')
            cam = cam.cpu().squeeze().numpy()

        return cam, torch.softmax(output, dim=1)