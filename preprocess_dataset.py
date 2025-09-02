import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path if it exists
if os.path.exists('src'):
    sys.path.append('src')


class DatasetPreprocessor:
    def __init__(self, input_dir='dataset', output_dir='dataset_processed'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = (224, 224)
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'classes': {}
        }

    def create_output_structure(self):
        """Create output directory structure"""
        print("📁 Creating output directory structure...")

        for split in ['Training', 'Testing']:
            split_path = self.input_dir / split
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        output_class_dir = self.output_dir / split / class_dir.name
                        output_class_dir.mkdir(parents=True, exist_ok=True)

    def enhance_contrast(self, image):
        """Apply CLAHE for contrast enhancement"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        return enhanced

    def remove_noise(self, image):
        """Denoise image"""
        return cv2.bilateralFilter(image, 9, 75, 75)

    def normalize_image(self, image):
        """Normalize image to [0, 255] range"""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def process_single_image(self, img_path, output_path):
        """Process a single image"""
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.array(Image.open(img_path).convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Step 1: Resize
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)

            # Step 2: Denoising
            image = self.remove_noise(image)

            # Step 3: Contrast enhancement
            image = self.enhance_contrast(image)

            # Step 4: Normalize
            image = self.normalize_image(image)

            # Save processed image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)

            return True, None

        except Exception as e:
            return False, str(e)

    def process_dataset(self):
        """Process entire dataset"""
        print(f"🚀 Starting preprocessing...")
        print(f"   Input: {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Target size: {self.target_size}")

        # Create output structure
        self.create_output_structure()

        # Process each split
        for split in ['Training', 'Testing']:
            split_path = self.input_dir / split
            if not split_path.exists():
                print(f"⚠️  {split_path} not found, skipping...")
                continue

            print(f"\n📂 Processing {split}...")

            # Process each class
            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue

                class_name = class_dir.name
                output_class_dir = self.output_dir / split / class_name

                # Get all image files
                image_files = list(class_dir.glob('*.jpg')) + \
                              list(class_dir.glob('*.jpeg')) + \
                              list(class_dir.glob('*.png')) + \
                              list(class_dir.glob('*.bmp'))

                if class_name not in self.stats['classes']:
                    self.stats['classes'][class_name] = {'training': 0, 'testing': 0}

                # Process each image
                with tqdm(total=len(image_files), desc=f"  {class_name}") as pbar:
                    for img_path in image_files:
                        output_path = output_class_dir / img_path.name

                        success, error = self.process_single_image(img_path, output_path)

                        if success:
                            self.stats['processed'] += 1
                            self.stats['classes'][class_name][split.lower()] += 1
                        else:
                            self.stats['failed'] += 1
                            print(f"    ⚠️ Failed: {img_path.name} - {error}")

                        pbar.update(1)

                self.stats['total_images'] += len(image_files)

        # Print summary
        self.print_summary()

        # Save report
        self.save_report()

    def print_summary(self):
        """Print preprocessing summary"""
        print("\n" + "=" * 60)
        print("📊 PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"✓ Total images found: {self.stats['total_images']}")
        print(f"✓ Successfully processed: {self.stats['processed']}")
        print(f"✗ Failed: {self.stats['failed']}")

        if self.stats['total_images'] > 0:
            success_rate = (self.stats['processed'] / self.stats['total_images']) * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")

        print("\n📈 Per-class statistics:")
        print("-" * 40)
        for class_name, counts in self.stats['classes'].items():
            total = counts['training'] + counts['testing']
            print(f"{class_name}:")
            print(f"  Training: {counts['training']} images")
            print(f"  Testing: {counts['testing']} images")
            print(f"  Total: {total} images")

    def save_report(self):
        """Save preprocessing report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'target_size': self.target_size,
            'statistics': self.stats
        }

        report_path = self.output_dir / 'preprocessing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Report saved to: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess Brain Tumor MRI Dataset')
    parser.add_argument('--input-dir', type=str, default='dataset',
                        help='Input dataset directory')
    parser.add_argument('--output-dir', type=str, default='dataset_processed',
                        help='Output directory for processed images')

    args = parser.parse_args()

    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"❌ Error: Input directory '{args.input_dir}' not found!")
        print("Please ensure your dataset is in the correct location.")
        return

    preprocessor = DatasetPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    preprocessor.process_dataset()
    print("\n✅ Preprocessing complete!")


if __name__ == '__main__':
    main()
