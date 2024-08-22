import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import numpy as np
import pandas as pd

from datasets import Dataset, Image, load_metric

label2id = {'cat': 0, 'dog': 1}
id2label = {0: 'cat', 1: 'dog'}

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

from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k")


from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

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


from transformers import ViTForImageClassification, TrainingArguments, Trainer
from transformers.models.vit.modeling_vit import ImageClassifierOutput


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ViTSCLForImageClassification(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = 0.2

    def forward(self, pixel_values=None, head_mask=None, labels=None):
        outputs = self.vit(pixel_values, head_mask=head_mask)
        sequence_output = outputs[0]
        hidden = sequence_output[:, 0, :]

        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_ce = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            cl_fct = SupConLoss()
            loss_cl = cl_fct(hidden, labels)

            loss = loss_ce + self.alpha * loss_cl

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = ViTSCLForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    # "google/vit-large-patch16-224-in21k",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=100,
    learning_rate=2e-5,
    save_total_limit=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

prediction_outputs = trainer.predict(dataset["target"])
print(prediction_outputs)
print(prediction_outputs[0])

logits = torch.from_numpy(np.array(prediction_outputs[0], dtype='float32'))
test_pred = torch.softmax(logits, dim=1).numpy()

result_output = pd.DataFrame(data={"id": test_image_id, "label": test_pred[:, 1]})
result_output.to_csv("./result/vit.csv", index=False, quoting=3)
