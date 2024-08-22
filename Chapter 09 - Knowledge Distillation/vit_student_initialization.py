from transformers import ViTModel, ViTConfig, TrainingArguments, Trainer
from transformers.models.vit.modeling_vit import ViTEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


def visualize_children(object, level: int=0):
    print(f"{'   ' * level}{level}- {type(object).__name__}")
    try:
        for child in object.children():
            visualize_children(child, level + 1)
    except:
        pass


visualize_children(vit)


def distill_vit(teacher_model):
    configuration = teacher_model.config.to_dict()
    configuration["num_hidden_layers"] //= 2
    configuration = ViTConfig.from_dict(configuration)
    student_model = type(teacher_model)(configuration)
    distill_vit_weights(teacher_model, student_model)
    return student_model


def distill_vit_weights(teacher, student):
    if isinstance(teacher, ViTModel) or type(teacher).__name__.startswith('ViTFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_vit_weights(teacher_part, student_part)

    elif isinstance(teacher, ViTEncoder):
        teacher_encoding_layers = [layer for layer in next(teacher.children())]
        student_encoding_layers = [layer for layer in next(student.children())]
        for i in range(len(student_encoding_layers)):
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    else:
        student.load_state_dict(teacher.state_dict())


student = distill_vit(vit)
print(student)

student.save_pretrained('vit-base-patch16-224-in21k-student')
