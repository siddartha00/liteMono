import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from monoLite.litemono import LiteMonoSystem
from dataset.kittiEigenSplit import KITTISelfSupDataModule


def get_args():
    parser = argparse.ArgumentParser(description="Train LiteMono on KITTI Self-Supervised Dataset")

    # --- Data settings ---
    parser.add_argument('--data_dir', type=str, default='data/splits',
                        help='Path to KITTI splits folder containing train/val/test csvs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--load_depth', action='store_true', help='Load ground truth depth if available')

    # --- Model settings ---
    parser.add_argument('--encoder_variant', type=str, default='base',
                        choices=['tiny', 'small', 'base', '8M'],
                        help='Variant of LiteMono encoder to use')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--decoder_channels', nargs='+', type=int, default=None,
                        help='List of decoder channels (default depends on encoder)')

    # --- Training settings ---
    parser.add_argument('--max_epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs (0 for CPU)')
    parser.add_argument('--precision', type=str, default='32',
                        help='Precision mode (e.g., "32", "16-mixed")')
    parser.add_argument('--log_dir', type=str, default='logs', help='Where to save logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Where to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Validation check interval (fraction of epoch or int steps)')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='Limit training batches per epoch (fraction or int for debugging)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Limit validation batches per epoch (fraction or int for debugging)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Accumulate gradients every N batches')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    print("========== LiteMono Training ==========")
    print(f"Data directory: {args.data_dir}")
    print(f"Encoder variant: {args.encoder_variant}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print("=======================================")

    # --- Data Module ---
    data_module = KITTISelfSupDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_depth=args.load_depth
    )

    # --- Model ---
    model = LiteMonoSystem(
        encoder_variant=args.encoder_variant,
        decoder_channels=args.decoder_channels,
        lr=args.lr
    )

    # --- Logging and Callbacks ---
    logger = TensorBoardLogger(save_dir=args.log_dir, name=f"LiteMono_{args.encoder_variant}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=f"{args.encoder_variant}" + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=1.0
    )

    # --- Training ---
    trainer.fit(model, datamodule=data_module)

    # --- Testing (optional) ---
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
