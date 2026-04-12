
import torch.nn.functional as F


def ce_loss(student_logits, labels):
    return F.cross_entropy(student_logits, labels)


def kd_loss(student_logits, teacher_logits, temperature):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)


def total_loss(student_logits, teacher_logits, labels, temperature, lambda_kd):
    loss_ce = ce_loss(student_logits, labels)
    loss_kd = kd_loss(student_logits, teacher_logits, temperature)
    loss    = loss_ce + lambda_kd * loss_kd
    return loss, loss_ce, loss_kd
