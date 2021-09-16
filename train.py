import argparse
import logging
import os
from pathlib import Path

import datasets
import torch
from datasets import (
    ClassLabel
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
)

import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str, default='bert-base-uncased')
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--fp16", type=bool, default=False)  # true for gpu

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str,
                        default='./exp_results',
                        # default=os.environ["SM_OUTPUT_DATA_DIR"],
                        )
    parser.add_argument("--model_dir", type=str,
                        default='./exp_results',
                        # default=os.environ["SM_MODEL_DIR"],
                        )
    parser.add_argument("--n_gpus", type=str, default="0")

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(Path('logs/train.log'))],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Create a new W&B Run
    wandb.init(project="hf-sexist-detector")

    # load datasets
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_dir = Path('./data')
    train_file = data_dir / 'train.csv'
    val_file = data_dir / 'val.csv'
    test_file = data_dir / 'test.csv'
    dataset = datasets.load_dataset("csv",
                                    data_files={'train': str(train_file),
                                                'val': str(val_file),
                                                'test': str(test_file),
                                                })


    # preprocess function, tokenizes text
    def preprocess_function(examples):
        to_ret = tokenizer(examples["Sentences"],
                           padding="max_length",
                           truncation=True)
        return to_ret


    class_label = ClassLabel(names_file='label_names.csv')
    dataset = dataset.map(
        lambda examples: {'label': examples['Label']},
        batched=True,
    )

    # preprocess dataset
    dataset = dataset.map(
        preprocess_function,
        batched=False,
    )

    # print size
    logger.info(f" loaded train dataset length is: {len(dataset['train'])}")
    logger.info(f" loaded val dataset length is: {len(dataset['val'])}")
    logger.info(f" loaded test dataset length is: {len(dataset['test'])}")


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                   y_true=labels,
                                                                   preds=preds,
                                                                   class_names=class_label.names,
                                                                   )})
        return {"accuracy": acc,
                "macro_f1": macro_f1,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "micro_f1": micro_f1,
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                }


    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id,
                                                               num_labels=len(class_label.names))

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        report_to='wandb',
        run_name='debug',
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=dataset['test'])

    # writes eval result to file which can be accessed later in s3 ouput
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # update the config for prediction
    trainer.model.config.label2id = class_label._str2int
    trainer.model.config.id2label = {i: s for i, s in enumerate(class_label._int2str)}

    # Saves the model to s3
    trainer.save_model(args.model_dir)

    # write incorrectly labeled instances to a wandb table
    # Create a W&B Table
    my_table = wandb.Table(columns=["sentence", "label", "prediction"])

    # write incorrectly labeled instances to a wandb table
    # Create a W&B Table
    my_table = wandb.Table(columns=["sentence", "label", "prediction"])

    # Get your image data and make predictions
    for split in ('train', 'val'):
        this_dataset = dataset[split]
        loader = torch.utils.data.DataLoader(this_dataset, batch_size=args.train_batch_size)

        for batch in tqdm(loader):
            input_ids = torch.stack(batch['input_ids'], dim=-1).to(trainer.model.device)
            attention_mask = torch.stack(batch['attention_mask'], dim=-1).to(trainer.model.device)
            token_type_ids = torch.stack(batch['token_type_ids'], dim=-1).to(trainer.model.device)
            labels = batch['label'].to(trainer.model.device)
            outputs = trainer.model(input_ids, attention_mask, token_type_ids)
            logits = outputs['logits']
            predictions = logits.argmax(1)

            # Add your image data and predictions to the W&B Table
            for idx, pred in enumerate(predictions):
                instance_input_ids = input_ids[idx]
                # get rid of PAD token
                instance_input_ids = instance_input_ids[torch.nonzero(instance_input_ids)].squeeze()
                tokens = tokenizer.convert_ids_to_tokens(instance_input_ids)
                sentence = tokenizer.convert_tokens_to_string(tokens)
                class_actual = class_label.int2str(labels[idx].item())
                class_predicted = class_label.int2str(pred.item())
                my_table.add_data(sentence,
                                  class_actual,
                                  class_predicted)
            torch.cuda.empty_cache()  # we get cuda memory errors without this...

        # Log your Table to W&B
        wandb.log({f"{split}/predictions": my_table})
