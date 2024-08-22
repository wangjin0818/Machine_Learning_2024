import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from transformers import AutoFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Image, load_metric
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

label2id = {'cat': 0, 'dog': 1}
id2label = {0: 'cat', 1: 'dog'}
num_labels = 2

train_image_list, train_label_list = [], []
train_path = os.path.join('e://', 'dataset', 'dog_vs_cat', 'train')
for i, filename in enumerate(os.listdir(train_path)):
    train_image_list.append(os.path.join(train_path, filename))
    train_label_list.append(label2id[filename.split('.')[0]])

test_image_list, test_image_id = [], []
test_path = os.path.join('e://', 'dataset', 'dog_vs_cat', 'test')
for i, filename in enumerate(os.listdir(test_path)):
    test_image_list.append(os.path.join(test_path, filename))
    test_image_id.append(filename.split('.')[0])

train = Dataset.from_dict({"image": train_image_list, "label": train_label_list}).cast_column("image", Image())
test = Dataset.from_dict({"image": test_image_list}).cast_column("image", Image())

dataset = train.train_test_split(0.2)
dataset["target"] = test
print(dataset["train"][0])

base_id = "google/vit-base-patch16-224-in21k"

# feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(base_id)

# preprocessing
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_transforms = Compose([RandomResizedCrop(feature_extractor.size), ToTensor(), normalize])


def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


dataset = dataset.with_transform(transforms)
test = dataset.with_transform(transforms)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()


# distiller
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


training_args = DistillationTrainingArguments(
    output_dir="./checkpoint",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    fp16=True,
    learning_rate=1e-4,
    seed=33,
    # logging & evaluation strategies
    logging_dir="./checkpoint/logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,

    # distilation parameters
    alpha=0.5,
    temperature=4.0
)

student_id = "vit-base-patch16-224-in21k-student"
teacher_id = "vit-base-patch16-224-in21k-finetuned"

# load model
teacher_model = ViTForImageClassification.from_pretrained(
    teacher_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# define student model
student_model = ViTForImageClassification.from_pretrained(
    student_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = DistillationTrainer(
    student_model,
    training_args,
    teacher_model=teacher_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
