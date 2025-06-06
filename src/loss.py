from typing import Literal

# KL - Kullback-Leibler Divergence
# CAKL - Confidence-Aware Kullback-Leibler Divergence
LossFunctionType = Literal["CrossEntropy", "KL", "CAKL", "Wasserstein"]


def get_loss_function(loss_type: LossFunctionType):
    return lambda x: x
