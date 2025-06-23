from typing import Literal

import torch
import torch.nn.functional as f

# KL - Kullback-Leibler Divergence
# CAKL - Confidence-Aware Kullback-Leibler Divergence

LossFunctionType = Literal["CrossEntropy", "KL", "CAKL", "Wasserstein", "CEKL", "CrossEntropyWithoutKD", "WagedKL"]


def get_loss_function(loss_type: LossFunctionType):
    if loss_type == "CrossEntropy":
        return cross_entropy
    elif loss_type == "CrossEntropyWithoutKD":
        return cross_entropy_without_kd
    elif loss_type == "KL":
        return KL_loss
    elif loss_type == "WagedKL":
        return waged_KL_loss
    elif loss_type == "Wasserstein":
        return wasserstein_loss
    elif loss_type == "CEKL":
        return cross_entropy_plus_KL
    elif loss_type == "CAKL":
        return CAKL_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def cross_entropy(
    student_logits: torch.Tensor, *, teacher_logits: torch.Tensor, **kwargs
) -> torch.Tensor:
    teacher_probs = f.softmax(teacher_logits, dim=-1)
    return f.cross_entropy(
        student_logits.reshape((-1, student_logits.size(-1))),
        teacher_probs.reshape((-1, teacher_probs.size(-1))),
    )


def cross_entropy_without_kd(
    student_logits: torch.Tensor, *, labels: torch.Tensor, **kwargs
) -> torch.Tensor:
    return f.cross_entropy(
        student_logits.reshape((-1, student_logits.size(-1))), labels.reshape(-1)
    )


def KL_loss(
    student_logits: torch.Tensor,
    *,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    teacher_probs = f.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = f.log_softmax(student_logits / temperature, dim=-1)
    return (
        f.kl_div(student_log_probs.reshape((-1, student_log_probs.size(-1))), teacher_probs.reshape((-1, teacher_probs.size(-1))), reduction="batchmean")
        * temperature ** 2
    )

def waged_KL_loss(
    student_logits: torch.Tensor,
    *,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    teacher_probs = f.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = f.log_softmax(student_logits / temperature, dim=-1)
    teacher_wages = torch.max(teacher_probs, dim=-1).values
    waged_dkl = (teacher_wages * f.kl_div(student_log_probs.reshape((-1, student_log_probs.size(-1))), teacher_probs.reshape((-1, teacher_probs.size(-1))),
                                            reduction="none").sum(dim=-1)).mean(dim=0)
    return waged_dkl * temperature ** 2

def CAKL_loss(student_logits: torch.Tensor,
    *,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    teacher_probs = f.softmax(teacher_logits / temperature, dim=-1)
    student_probs = f.softmax(student_logits / temperature, dim=-1)
    selected_idcs = teacher_probs.argmax(dim=-1)
    selected_probs = torch.gather(teacher_probs, dim=-1, index=selected_idcs.unsqueeze(-1)).squeeze(-1)
    gamma = selected_probs.mean(dim=-1).mean(dim=0)
    return (gamma *  f.kl_div(student_probs.log().reshape((-1, student_probs.size(-1))), teacher_probs.reshape((-1, teacher_probs.size(-1))), 
                            reduction="batchmean") + (1 - gamma) * f.kl_div(teacher_probs.log().reshape((-1, teacher_probs.size(-1))),
                                                                            student_probs.reshape((-1, student_probs.size(-1))),
                                                                            reduction="batchmean")
                            )
    
def wasserstein_loss(student_logits: torch.Tensor, *, teacher_logits: torch.Tensor, 
                    temperature: float=1.0, **kwargs) -> torch.Tensor:
    teacher_probs = f.softmax(teacher_logits / temperature, dim=-1)
    student_probs = f.softmax(student_logits / temperature, dim=-1)
    teacher_probs_sorted = torch.sort(teacher_probs, dim=-1, descending=True)[0]
    student_probs_sorted = torch.sort(student_probs, dim=-1, descending=True)[0]
    return torch.abs(teacher_probs_sorted.reshape((-1, teacher_probs_sorted.size(-1))) - student_probs_sorted.reshape((-1, student_probs_sorted.size(-1)))).sum(dim=-1).mean(dim=0)

def cross_entropy_plus_KL(student_logits: torch.Tensor, *, teacher_logits: torch.Tensor,
                        temperature: float=1, lbda: float=1, **kwargs) -> torch.Tensor:
    return f.cross_entropy(student_logits, teacher_logits) + lbda * KL_loss(teacher_logits, student_logits, temperature)

if __name__ == "__main__":
    # Tests
    batch_size, vocab_size = 8, 200
    torch.manual_seed(42)
    teacher_logits = torch.randn(batch_size, vocab_size, dtype=torch.float16)
    student_logits = torch.randn(batch_size, vocab_size, dtype=torch.float16)
    print(get_loss_function("Wasserstein")(student_logits, teacher_logits=teacher_logits))
