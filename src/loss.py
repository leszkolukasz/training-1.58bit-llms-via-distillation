from typing import Literal
import torch 
import torch.nn.functional as f

# KL - Kullback-Leibler Divergence
# CAKL - Confidence-Aware Kullback-Leibler Divergence
LossFunctionType = Literal["CrossEntropy", "KL", "CAKL", "Wasserstein"]


def get_loss_function(loss_type: LossFunctionType):
    if loss_type == "CrossEntropy":
        return cross_entropy
    elif loss_type == "KL":
        return KL_loss
    elif loss_type == "CAKL":
        return CAKL_loss
    elif loss_type == "Wasserstein":
        return wasserstein_loss
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def cross_entropy(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    return f.cross_entropy(teacher_logits, student_logits)

def KL_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    teacher_log_probs = f.log_softmax(teacher_logits / temperature)
    student_log_probs = f.log_softmax(student_logits / temperature)
    return f.kl_div(teacher_log_probs, student_log_probs, reduce="batchmean") * temperature ** 2

def CAKL_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
    teacher_log_probs = f.log_softmax(teacher_logits / temperature)
    student_log_probs = f.log_softmax(student_logits / temperature)
    teacher_log_wages = torch.max(teacher_log_probs, dim=-1).values
    waged_dkl = (teacher_log_wages * f.kl_div(teacher_log_probs, student_log_probs, reduction="none").sum(dim=-1)).mean(dim=0)
    return waged_dkl * temperature ** 2
    
def wasserstein_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, 
                    temperature: float=1.0) -> torch.Tensor:
    teacher_cdf = f.softmax(teacher_logits / temperature).cumsum(dim=-1)
    student_cdf = f.softmax(student_logits / temperature).cumsum(dim=-1)
    return torch.abs(teacher_cdf - student_cdf).sum(dim=-1).mean(dim=0)
    
if __name__ == "__main__":
    # Tests
    batch_size, vocab_size = 8, 200
    teacher_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    student_logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    print(get_loss_function("KL")(teacher_logits, student_logits))
    