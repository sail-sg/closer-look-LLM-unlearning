import copy

import deepspeed
from transformers import Trainer

from .losses import get_loss


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('loss_type')
        self.ref_model = kwargs.pop('ref_model')

        # the coefficient of each part in the loss function. This is used in ablation study.
        self.forget_coeff = kwargs.pop('forget_coeff')
        self.regularization_coeff = kwargs.pop('regularization_coeff')
        # beta for NPO/DPO/RS
        self.beta = kwargs.pop('beta')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        self.ref_model = self.e_prepare_deepspeed(self.ref_model)

    def compute_loss(self, model, inputs, return_outputs=False):

        forget_loss, regularization_loss = get_loss(model, self.ref_model, inputs, self.loss_type, self.beta)
        loss = self.forget_coeff * forget_loss + self.regularization_coeff * regularization_loss

        return (loss, None) if return_outputs else loss

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        # set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model
