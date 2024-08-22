import os
import torch
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


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
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
    per_device_train_batch_size=24,
    per_device_eval_batch_size=48,
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
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

model.save_pretrained('vit-base-patch16-224-in21k-finetuned')

prediction_outputs = trainer.predict(dataset["target"])
print(prediction_outputs)
print(prediction_outputs[0])

logits = torch.from_numpy(np.array(prediction_outputs[0], dtype='float32'))
test_pred = torch.softmax(logits, dim=1).numpy()

result_output = pd.DataFrame(data={"id": test_image_id, "label": test_pred[:, 1]})
result_output.to_csv("./result/vit.csv", index=False, quoting=3)
