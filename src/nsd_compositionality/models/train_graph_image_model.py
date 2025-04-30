import os

import hydra
import mlflow
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from transformers import CLIPConfig, CLIPProcessor, GraphormerConfig, Trainer, TrainingArguments

from nsd_compositionality.data.preprocess_graphormer import GraphCLIPDataCollator, preprocess_item
from nsd_compositionality.models.modeling_graph_image_model import GraphCLIPModel


def preprocess_dataset(dataset, cfg):
    # Initialize CLIP processor for image and text
    clip_processor = CLIPProcessor.from_pretrained(cfg.model.pretrained_model_name_or_path)

    # Preprocess the dataset
    def preprocess_function(example):
        # Preprocess image and text
        processed = clip_processor(
            text=example["sentences_raw"],
            images=[Image.open(os.path.join(cfg.data.image_base_path, img)) for img in example["filepath"]],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        # Preprocess graph input
        graph_input = [preprocess_item(ex) for ex in example["graph_input"]]
        processed["graph_input"] = graph_input
        return processed

    # Apply preprocessing
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=cfg.data.num_proc,
        batch_size=cfg.data.batch_size,
    )
    return dataset


@hydra.main(config_path="../../../configs/model", config_name="train_graph_image_model")
def train_graph_image_model(cfg: DictConfig):
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(cfg)

        # Load preprocessed data
        dataset = load_dataset(
            cfg.data.hf_dataset_identifier,
            split=cfg.data.split,
        )
        # Only keep the graph type column specified by model_type_graph_base in the config, remove all other
        # columns that contain "_graphs"
        target_graph_column = cfg.model.model_type_graph_base + "_" if cfg.model.model_type_graph_base else ""
        target_graph_column = target_graph_column + cfg.model.model_type + "_graphs"
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col.endswith("_graphs") and col != target_graph_column]
        )
        # Rename the target graph column to "input_graphs"
        dataset = dataset.rename_column(target_graph_column, "graph_input")
        len_dataset_pre = len(dataset)
        # Filter out data with empty graphs: num_nodes == 0 or edge_index == [[], []]
        dataset = dataset.filter(
            lambda x: x["graph_input"]["num_nodes"] > 0 or x["graph_input"]["edge_index"] != [[], []]
        )
        # Log the number of samples in the dataset after filtering
        mlflow.log_metric("empty_graphs_ratio", 1 - len(dataset) / len_dataset_pre)
        # Make sure the dataset is shuffled
        dataset = dataset.shuffle(seed=cfg.data.seed)
        if cfg.data.n_samples > 0:
            dataset = dataset.select(range(cfg.data.n_samples))

        # Preprocess the dataset
        dataset = preprocess_dataset(dataset, cfg)
        # Push it to the huggingface hub
        if cfg.data.push_to_hub:
            dataset.push_to_hub(cfg.data.hf_dataset_identifier_processed + "_" + target_graph_column)

        # Set a validation set aside
        dataset = dataset.train_test_split(test_size=cfg.data.validation_split, seed=cfg.data.seed)
        # Not to be confused with the train/test split from the load_dataset function
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        # Create a configuration for the GraphCLIP Model
        graphormer_config = GraphormerConfig()
        config = CLIPConfig(
            graph_config=graphormer_config,
            graph_pair_type=cfg.model.model_type,
            pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        )

        # Initialize the model
        model = GraphCLIPModel(config)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            save_strategy="steps",
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.batch_size,
            num_train_epochs=cfg.training.epochs,
            weight_decay=cfg.training.weight_decay,
            logging_dir=os.path.join(cfg.output_dir, "logs"),
            logging_steps=cfg.training.logging_steps,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=True,
            report_to=["mlflow"],  # Integrate with MLflow
        )

        # Define Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=GraphCLIPDataCollator(on_the_fly_processing=False),
        )

        # Train the model
        trainer.train()

        # Define the model save path
        model_save_path = os.path.join(cfg.output_dir, "graph_clip_model" + "_" + target_graph_column)

        # Push the model to the huggingface hub
        model.push_to_hub(cfg.model.huggingface_hub_model_id + "_" + target_graph_column)

        # And save it locally
        trainer.save_model(model_save_path)


if __name__ == "__main__":
    train_graph_image_model()
