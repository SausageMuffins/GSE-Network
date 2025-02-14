import os
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from model import GSENet  # Import GSENet model
from dataset import GSEDataLoader, pad_collate_fn  # Import dataset loader
from losses import GSENetSTFTLoss  # Import custom STFT loss
from google.cloud import storage  # For Google Cloud Storage (optional)
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint callback

class GSENetTrainer(pl.LightningModule):
    """
    PyTorch Lightning training module for GSENet.
    Handles training, validation, and checkpointing.
    """

    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = GSENetSTFTLoss()  # STFT-based loss function

    def forward(self, ref_mic, beamformed):
        return self.model(ref_mic, beamformed)

    def training_step(self, batch, batch_idx):
        ref_mic, beamformed, clean_speech = batch
        enhanced_wav = self(ref_mic, beamformed)
        loss = self.loss_fn(enhanced_wav, clean_speech)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ref_mic, beamformed, clean_speech = batch
        enhanced_wav = self(ref_mic, beamformed)
        loss = self.loss_fn(enhanced_wav, clean_speech)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

def upload_checkpoint_to_gcs(local_path, gcs_bucket, gcs_path):
    """
    Uploads a local checkpoint file to Google Cloud Storage (GCS).
    """
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Checkpoint uploaded to gs://{gcs_bucket}/{gcs_path}")

def main(args):
    """
    Main function to train GSENet using PyTorch Lightning.
    """
    # Load dataset
    train_loader = DataLoader(
        GSEDataLoader(args.data_path, split="train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        GSEDataLoader(args.data_path, split="val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_collate_fn,
    )

    # Initialize model
    model = GSENet()

    # Setup checkpoint callback to save the best model based on the lowest validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filename='gsenet_model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=20,
        default_root_dir=args.checkpoint_dir,
        callbacks=[checkpoint_callback],
    )

    # Train model
    trainer.fit(GSENetTrainer(model), train_loader, val_loader)

    # Retrieve the best checkpoint path saved by the callback
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path == "":
        # Fallback if for some reason the best model wasn't saved during training
        best_model_path = os.path.join(args.checkpoint_dir, "gsenet_model.ckpt")
        trainer.save_checkpoint(best_model_path)
    print(f"Best model saved at {best_model_path}")

    # Upload best checkpoint to GCS if specified
    if args.gcs_bucket:
        upload_checkpoint_to_gcs(best_model_path, args.gcs_bucket, "checkpoints/gsenet_model.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Guided Speech Enhancement Network (GSENet)")
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--gcs_bucket", type=str, default=None, help="Google Cloud Storage bucket for checkpoint storage (optional)")

    args = parser.parse_args()
    main(args)
