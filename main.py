import os
import sys
import argparse
import torch
import wandb
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config import Config
from src.utils.logger import Logger
from src.preprocessing.data_loader import get_data_loaders
from src.models.classifier import BrainTumorClassifier
from src.models.segmenter import AttentionUNet
from src.models.hybrid_pipeline import HybridBrainTumorModel
from src.training.train_classifier import ClassifierTrainer
from src.training.train_segmenter import SegmenterTrainer
from src.evaluation.metrics import MetricsCalculator
from src.inference.predict import BrainTumorPredictor
from src.inference.visualize_results import Visualizer
from src.inference.report_generator import ReportGenerator


def train_pipeline(args):
    """
    Complete training pipeline
    """
    # Initialize
    config = Config()
    logger = Logger(log_dir=str(config.LOG_DIR))

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="brain-tumor-detection",
            config={
                'learning_rate_clf': config.LEARNING_RATE_CLF,
                'learning_rate_seg': config.LEARNING_RATE_SEG,
                'batch_size': config.BATCH_SIZE,
                'epochs_clf': config.EPOCHS_CLF,
                'epochs_seg': config.EPOCHS_SEG,
                'architecture_clf': config.CLF_BACKBONE,
                'architecture_seg': config.SEG_BACKBONE
            }
        )

    logger.info("Starting Brain Tumor Detection Pipeline...")
    logger.info(f"Configuration: {vars(config)}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Phase 1: Train Classifier
    if args.train_classifier:
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 1: Training Classifier")
        logger.info("=" * 50)

        classifier = BrainTumorClassifier(
            num_classes=config.NUM_CLASSES,
            backbone=config.CLF_BACKBONE,
            dropout_rate=config.DROPOUT_RATE
        )

        clf_trainer = ClassifierTrainer(classifier, config, logger)
        best_clf_state = clf_trainer.train(
            train_loader,
            val_loader,
            config.EPOCHS_CLF
        )

        logger.info(f"Best classifier accuracy: {clf_trainer.best_val_acc:.4f}")

    # Phase 2: Train Segmenter
    if args.train_segmenter:
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2: Training Segmenter")
        logger.info("=" * 50)

        segmenter = AttentionUNet(in_channels=3, out_channels=1)

        seg_trainer = SegmenterTrainer(segmenter, config, logger)
        best_seg_state = seg_trainer.train(
            train_loader,
            val_loader,
            config.EPOCHS_SEG
        )

        logger.info(f"Best segmentation Dice: {seg_trainer.best_dice:.4f}")

    # Phase 3: Train Hybrid Model (Optional)
    if args.train_hybrid:
        logger.info("\n" + "=" * 50)
        logger.info("PHASE 3: Training Hybrid Model")
        logger.info("=" * 50)

        hybrid_model = HybridBrainTumorModel(
            num_classes=config.NUM_CLASSES,
            backbone=config.CLF_BACKBONE
        )

        # Load pretrained weights if available
        if (config.MODEL_DIR / 'best_classifier.pth').exists():
            classifier_state = torch.load(config.MODEL_DIR / 'best_classifier.pth')
            hybrid_model.classifier.load_state_dict(classifier_state, strict=False)
            logger.info("Loaded pretrained classifier weights")

        if (config.MODEL_DIR / 'best_segmenter.pth').exists():
            segmenter_state = torch.load(config.MODEL_DIR / 'best_segmenter.pth')
            hybrid_model.segmenter.load_state_dict(segmenter_state, strict=False)
            logger.info("Loaded pretrained segmenter weights")

        # Fine-tune hybrid model
        # ... (hybrid training code similar to above)

    # Phase 4: Evaluation
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: Final Evaluation")
    logger.info("=" * 50)

    evaluate_models(test_loader, config, logger)

    logger.info("\nTraining pipeline completed successfully!")

    if args.use_wandb:
        wandb.finish()


def evaluate_models(test_loader, config, logger):
    """
    Evaluate trained models on test set
    """
    device = torch.device(config.DEVICE)
    metrics_calc = MetricsCalculator(num_classes=config.NUM_CLASSES)

    # Load best models
    if (config.MODEL_DIR / 'best_classifier.pth').exists():
        classifier = BrainTumorClassifier(
            num_classes=config.NUM_CLASSES,
            backbone=config.CLF_BACKBONE
        )
        classifier.load_state_dict(
            torch.load(config.MODEL_DIR / 'best_classifier.pth')
        )
        classifier.to(device)
        classifier.eval()

        # Evaluate classifier
        logger.info("Evaluating classifier...")
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data in test_loader:
                if len(data) == 3:
                    images, labels, _ = data
                else:
                    images, labels = data

                images = images.to(device)
                labels = labels.to(device)

                outputs, _ = classifier(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics_calc.update(all_preds, all_labels, all_probs)
        clf_metrics = metrics_calc.compute_classification_metrics()

        logger.info("Classification Results:")
        logger.info(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {clf_metrics['f1']:.4f}")
        logger.info(f"  AUROC: {clf_metrics.get('auroc', 0):.4f}")
        logger.info("\nClassification Report:")
        logger.info(clf_metrics['classification_report'])


def inference_pipeline(args):
    """
    Run inference on single image or directory
    """
    config = Config()
    logger = Logger()

    logger.info("Starting inference pipeline...")

    # Check model files
    clf_path = config.MODEL_DIR / 'best_classifier.pth'
    seg_path = config.MODEL_DIR / 'best_segmenter.pth'

    if not clf_path.exists() or not seg_path.exists():
        logger.error("Model files not found! Please train models first.")
        return

    # Initialize predictor
    predictor = BrainTumorPredictor(clf_path, seg_path, config)
    visualizer = Visualizer(config)
    report_gen = ReportGenerator()

    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # Single image inference
        logger.info(f"Processing: {input_path}")

        # Run prediction
        results = predictor.predict(str(input_path))

        # Generate visualizations
        from PIL import Image
        image = Image.open(input_path)
        vis_path = output_dir / f"{input_path.stem}_visualization.png"
        visualizer.visualize_prediction(image, results, save_path=str(vis_path))
        logger.info(f"Visualization saved to: {vis_path}")

        # Generate reports
        pdf_path = output_dir / f"{input_path.stem}_report.pdf"
        patient_info = {'id': input_path.stem, 'age': 'Unknown'}
        report_gen.generate_pdf_report(results, patient_info, str(pdf_path))
        logger.info(f"PDF report saved to: {pdf_path}")

        json_path = output_dir / f"{input_path.stem}_results.json"
        report_gen.generate_json_report(results, str(json_path))
        logger.info(f"JSON results saved to: {json_path}")

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Tumor Detected: {results['tumor_detected']}")
        if results['tumor_detected']:
            logger.info(f"Tumor Type: {results['tumor_type']}")
            logger.info(f"Confidence: {results['confidence']:.2%}")
            logger.info(f"Volume: {results['volume']['volume_cm3']:.2f} cm³")
            logger.info(f"Staging: {results['staging']['grade']}")
            logger.info(f"Malignancy: {results['staging']['malignancy']}")

    elif input_path.is_dir():
        # Batch inference
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        logger.info(f"Found {len(image_files)} images to process")

        for img_path in image_files:
            logger.info(f"Processing: {img_path.name}")

            try:
                results = predictor.predict(str(img_path))

                # Save results
                json_path = output_dir / f"{img_path.stem}_results.json"
                report_gen.generate_json_report(results, str(json_path))

                # Generate visualization
                image = Image.open(img_path)
                vis_path = output_dir / f"{img_path.stem}_visualization.png"
                visualizer.visualize_prediction(image, results, save_path=str(vis_path))

            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {str(e)}")
                continue

    logger.info("\nInference pipeline completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection & Analysis Pipeline'
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Training mode
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--train-classifier', action='store_true',
                              help='Train classification model')
    train_parser.add_argument('--train-segmenter', action='store_true',
                              help='Train segmentation model')
    train_parser.add_argument('--train-hybrid', action='store_true',
                              help='Train hybrid model')
    train_parser.add_argument('--use-wandb', action='store_true',
                              help='Use Weights & Biases for logging')

    # Inference mode
    infer_parser = subparsers.add_parser('predict', help='Run inference')
    infer_parser.add_argument('--input', type=str, required=True,
                              help='Input image or directory')
    infer_parser.add_argument('--output-dir', type=str, default='./output',
                              help='Output directory for results')

    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--test-dir', type=str,
                             help='Test dataset directory')

    args = parser.parse_args()

    if args.mode == 'train':
        train_pipeline(args)
    elif args.mode == 'predict':
        inference_pipeline(args)
    elif args.mode == 'evaluate':
        config = Config()
        logger = Logger()
        _, _, test_loader = get_data_loaders(config)
        evaluate_models(test_loader, config, logger)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()