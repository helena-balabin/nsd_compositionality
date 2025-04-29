import os

import hydra
import mlflow
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import CLIPConfig, GraphormerConfig, Trainer, TrainingArguments

# from nsd_compositionality.data.preprocess_graphormer import GraphormerDataCollator, preprocess_item
from nsd_compositionality.models.modeling_graph_image_model import GraphCLIPModel


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
        # Make sure the dataset is shuffled
        dataset = dataset.shuffle(seed=cfg.data.seed)

        # Set a validation set aside
        dataset = dataset.train_test_split(test_size=cfg.data.validation_split, seed=cfg.data.seed)
        # Not to be confused with the train/test split from the load_dataset function
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        # TODO preprocess the dataset with GraphormerDataCollator on the fly, similar to HF example,
        # make sure all the features are preserved
        # It is also possible to apply this preprocessing on the fly, in the DataCollator's parameters
        # (by setting on_the_fly_processing to True):
        # not all datasets are as small as ogbg-molhiv, and for large graphs, it might be too costly to
        # store all the preprocessed data beforehand.

        # Create a configuration for the GraphCLIP Model
        graphormer_config = GraphormerConfig()
        config = CLIPConfig(
            graph_config=graphormer_config,
            graph_pair_type="image",
            pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
        )

        # Initialize the model
        model = GraphCLIPModel(config)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            evaluation_strategy="steps",
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
        )

        # Train the model
        trainer.train()

        # Save the trained model
        model_save_path = os.path.join(cfg.output_dir, "graph_clip_model")
        trainer.save_model(model_save_path)
        mlflow.log_artifact(model_save_path)

        # Push the model to the huggingface hub
        model.push_to_hub(cfg.model.huggingface_hub_model_id)

        print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_graph_image_model()
