# Imports
import os
import pathlib
from shutil import copy2
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

import evaluate
import numpy as np
import torch

import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)


from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
wandb.init(
        project="hackathon-os", 
        name="train_5_t1",
        config={
            "model": "MCG-NJU/videomae-base",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "num_epochs": 1500,
            "warmup_ratio":0.1,
            "logging_steps":10,
            # Add more hyperparameters as needed
        }
    )

os.environ('CUDA_VISIBLE_DEVICES') = '0,1'

######################################################################################
######################          DEFINE PATHS HERE      ###############################
######################################################################################

dataset_root_path = './Datav1'
model_ckpt = "MCG-NJU/videomae-base"
batch_size = 8

######################################################################################
######################################################################################
######################################################################################

all_video_file_paths = dataset_root_path
# Get all the subdirectories in the training folder
class_labels = sorted([d.name for d in os.scandir(os.path.join(dataset_root_path, "train")) if d.is_dir()])
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}


image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
device = "cuda" #if torch.cuda.is_available() else "cpu"  ## Force to train on GPU
model.to(device)

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps


# Training dataset transformations.
train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)

# Training dataset. Yes keep this Ucf101 I have done the appropiate transformations for this to work
train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

# Validation and evaluation datasets' transformations.
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

# Validation and evaluation datasets.
val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)



model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-ucf101-subset"
######### You will need to find the hyperparameters 
######### out yourself, use ray to find them. But first train and see the acc. 
######### From what I read you will need around 1500-2000 epochs
num_epochs = 5
wandb.watch(model, log_freq=100)
wandb_logger = WandbLogger(log_model='all')
args = TrainingArguments(
    new_model_name,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # push_to_hub=True,
    max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
    report_to="wandb",
)


metric = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    # Additional metrics can be added as needed
    # For example, you can add more classification metrics or any custom metrics

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }




def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



    # Create Trainer instance with WandBCallback
trainer = Trainer(
        model,
        args,
        device_map = 'auto', 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        
    )

    

# ... (previous code)

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ... (remaining imports)

# Print the number of samples found in each dataset
# print(f"Number of samples in training dataset: {len(train_dataset)}")
# print(f"Number of samples in validation dataset: {len(val_dataset)}")
# print(f"Number of samples in test dataset: {len(test_dataset)}")

trainer.train()
# # Training loop
# for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
#     # Training
#     trainer.train()
    
#     # Validation
#     eval_results = trainer.evaluate()
#     print("***** Evaluation Results *****")
#     for key, value in eval_results.items():
#         print(f"{key}: {value}")

# # Save the model
trainer.save_model()
print("Model saved to:", 'out')

wandb.finish()


# Additional information during and after training
print("***** Training Finished *****")
print(f"Total training time: {trainer.total_training_time:.2f} seconds")

# Evaluation on the validation set
eval_results = trainer.evaluate()
print("***** Evaluation Results *****")
for key, value in eval_results.items():
    print(f"{key}: {value}")

# Save the model
trainer.save_model()
print("Model saved to:", 'out')

