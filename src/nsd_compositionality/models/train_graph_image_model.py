import os

import hydra
import mlflow
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from transformers import (
    CLIPConfig,
    CLIPProcessor,
    EarlyStoppingCallback,
    GraphormerConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

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


class ThreePhaseTrainingCallback(TrainerCallback):
    def __init__(self, model, total_epochs, phase_epochs, cfg):
        self.model = model
        self.total_epochs = total_epochs
        self.phase_epochs = phase_epochs
        self.cfg = cfg

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = state.epoch

        # Phase 1: Train only the graph component
        if current_epoch < self.phase_epochs:
            self.model.freeze_layers(freeze_vision=True, freeze_text=True, freeze_graph=False)

        # Phase 2: Unfreeze the last third of the text or image encoder
        elif current_epoch < 2 * self.phase_epochs:
            if self.cfg.model.model_type == "text":
                self.model.unfreeze_partial_layers(
                    "text",
                    num_layers=self.model.text_model.config.num_hidden_layers // 3,
                )
            elif self.cfg.model.model_type == "image":
                self.model.unfreeze_partial_layers(
                    "vision",
                    num_layers=self.model.vision_model.config.num_hidden_layers // 3,
                )

        # Phase 3: Unfreeze all parameters of the text or image encoder
        else:
            if self.cfg.model.model_type == "text":
                self.model.unfreeze_partial_layers(
                    "text",
                    num_layers=self.model.text_model.config.num_hidden_layers,
                )
            elif self.cfg.model.model_type == "image":
                self.model.unfreeze_partial_layers(
                    "vision",
                    num_layers=self.model.vision_model.config.num_hidden_layers,
                )


@hydra.main(config_path="../../../configs/model", config_name="train_graph_image_model")
def train_graph_image_model(cfg: DictConfig):
    # Load environment variables
    load_dotenv("../../../.env")
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Define the target graph column based on the model type
    target_graph_column = cfg.model.model_type_graph_base + "_" if cfg.model.model_type_graph_base else ""
    target_graph_column = target_graph_column + cfg.model.model_type + "_graphs"

    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(cfg)

        # Load preprocessed data
        if cfg.data.use_preprocessed:
            dataset = load_dataset(
                cfg.data.hf_dataset_identifier_processed + "_" + target_graph_column,
                split=cfg.data.split,
                cache_dir=cfg.data.cache_dir,
            )
        else:
            # Apply some preprocessing to the dataset
            dataset = load_dataset(
                cfg.data.hf_dataset_identifier,
                split=cfg.data.split,
                cache_dir=cfg.data.cache_dir,
            )
            # Get rid of duplicates using the "filepath" column we deal with images
            if cfg.model.model_type == "image":
                df = dataset.to_pandas()
                # Drop exact duplicate filepaths, keeping the first occurrence
                df_dedup = df.drop_duplicates(subset=["filepath"], keep="first")
                # Re-create a Dataset (will re-generate a new index column by default)
                dataset = Dataset.from_pandas(df_dedup)

            # Only keep the graph type column specified by model_type_graph_base in the config, remove all other
            # columns that contain "_graphs"
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
            mlflow.log_param("empty_graphs_ratio", 1 - len(dataset) / len_dataset_pre)
            # Make sure the dataset is shuffled
            dataset = dataset.shuffle(seed=cfg.data.seed)

            # Preprocess the dataset
            dataset = preprocess_dataset(dataset, cfg)
            # Push it to the huggingface hub
            if cfg.data.push_to_hub:
                dataset.push_to_hub(cfg.data.hf_dataset_identifier_processed + "_" + target_graph_column)

        # For debug purposes, limit the number of samples
        if cfg.data.n_samples > 0:
            dataset = dataset.select(range(cfg.data.n_samples))

        # Log the number of samples in the dataset
        mlflow.log_param("len_dataset", len(dataset))

        # Set a validation set aside
        dataset = dataset.train_test_split(test_size=cfg.data.validation_split, seed=cfg.data.seed)
        # Not to be confused with the train/test split from the load_dataset function
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        # Create a configuration for the GraphCLIP Model
        if cfg.model.graphormer_size == "small":
            graphormer_config = GraphormerConfig(
                hidden_size=512,
                embedding_dim=512,
                ffn_embedding_dim=512,
                num_hidden_layers=6,
                dropout=cfg.model.dropout,
            )
        else:
            graphormer_config = GraphormerConfig(
                dropout=cfg.model.dropout,
            )

        config = CLIPConfig(
            graph_config=graphormer_config,
            graph_pair_type=cfg.model.model_type,
            pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        )

        # Initialize the model
        model = GraphCLIPModel(config)

        # Define the optimizer
        optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

        training_steps_per_epoch = (
            len(train_dataset) // (cfg.training.batch_size * cfg.training.gradient_accumulation_steps) + 1
        )
        # Define the MultiStepLR scheduler with milestones corresponding to phase boundaries
        scheduler = MultiStepLR(
            optimizer,
            milestones=[
                cfg.training.epochs // 3 * training_steps_per_epoch,
                2 * cfg.training.epochs // 3 * training_steps_per_epoch,
            ],
            gamma=cfg.training.lr_gamma,
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            save_strategy="steps",
            save_steps=cfg.training.save_steps,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.batch_size,
            num_train_epochs=cfg.training.epochs,
            weight_decay=cfg.training.weight_decay,
            logging_dir=os.path.join(cfg.output_dir, "logs"),
            logging_steps=cfg.training.logging_steps,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            lr_scheduler_type="constant",
            report_to=["mlflow"],  # Integrate with MLflow
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=GraphCLIPDataCollator(on_the_fly_processing=False),
            optimizers=(optimizer, scheduler),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience),
                ThreePhaseTrainingCallback(model, cfg.training.epochs, cfg.training.epochs // 3, cfg),
            ],
        )

        # Train the model
        trainer.train()

        # Save and push the final model
        model_save_path = os.path.join(cfg.output_dir, "graph_clip_model" + "_" + target_graph_column)
        model.push_to_hub(cfg.model.huggingface_hub_model_id + "_" + target_graph_column)
        trainer.save_model(model_save_path)


if __name__ == "__main__":
    train_graph_image_model()
