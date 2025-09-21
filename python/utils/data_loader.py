"""
محمل البيانات لمجموعة بيانات أمراض الجلد
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

class SkinDiseaseDataset(Dataset):
    """
    مجموعة بيانات أمراض الجلد
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        image_dir: Union[str, Path],
        transform: Optional[A.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_to_idx = class_to_idx or self._create_class_mapping()
        
        # التحقق من وجود الصور
        self._validate_images()
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """إنشاء خريطة الفئات"""
        unique_classes = sorted(self.data_df['diagnosis'].unique())
        return {cls: idx for idx, cls in enumerate(unique_classes)}
    
    def _validate_images(self):
        """التحقق من وجود الصور"""
        missing_images = []
        for idx, row in self.data_df.iterrows():
            image_path = self.image_dir / row['image_id']
            if not image_path.exists():
                missing_images.append(str(image_path))
        
        if missing_images:
            print(f"تحذير: {len(missing_images)} صورة مفقودة")
            if len(missing_images) <= 10:
                for img in missing_images:
                    print(f"  - {img}")
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data_df.iloc[idx]
        
        # تحميل الصورة
        image_path = self.image_dir / row['image_id']
        image = self._load_image(image_path)
        
        # الحصول على التسمية
        label = self.class_to_idx[row['diagnosis']]
        
        # تطبيق التحويلات
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """تحميل الصورة"""
        try:
            # تحميل باستخدام OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                # محاولة تحميل باستخدام PIL
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
            else:
                # تحويل من BGR إلى RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            print(f"خطأ في تحميل الصورة {image_path}: {e}")
            # إرجاع صورة فارغة
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """الحصول على توزيع الفئات"""
        return self.data_df['diagnosis'].value_counts().to_dict()
    
    def get_class_weights(self) -> torch.Tensor:
        """حساب أوزان الفئات للتوازن"""
        labels = [self.class_to_idx[diagnosis] for diagnosis in self.data_df['diagnosis']]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights)

class SkinDiseaseDataModule(pl.LightningDataModule):
    """
    وحدة البيانات لـ PyTorch Lightning
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.data_path = Path(config['data']['data_path'])
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.pin_memory = config['data']['pin_memory']
        self.image_size = config['data']['image_size']
        
        # متغيرات البيانات
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_to_idx = None
        self.class_weights = None
    
    def prepare_data(self):
        """تحضير البيانات (تحميل وتنظيف)"""
        print("تحضير البيانات...")
        
        # البحث عن ملفات البيانات
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"لم يتم العثور على ملفات CSV في {self.data_path}")
        
        # تحميل البيانات
        self.data_df = pd.read_csv(csv_files[0])
        
        # تنظيف البيانات
        self._clean_data()
        
        # إنشاء خريطة الفئات
        self.class_to_idx = self._create_class_mapping()
        
        print(f"تم تحميل {len(self.data_df)} عينة")
        print(f"عدد الفئات: {len(self.class_to_idx)}")
        print("توزيع الفئات:")
        for cls, count in self.data_df['diagnosis'].value_counts().items():
            print(f"  {cls}: {count}")
    
    def _clean_data(self):
        """تنظيف البيانات"""
        # إزالة الصفوف الفارغة
        self.data_df = self.data_df.dropna(subset=['image_id', 'diagnosis'])
        
        # توحيد أسماء الأمراض
        diagnosis_mapping = {
            'mel': 'melanoma',
            'nv': 'nevus',
            'bcc': 'basal_cell_carcinoma',
            'akiec': 'actinic_keratosis',
            'bkl': 'benign_keratosis',
            'df': 'dermatofibroma',
            'vasc': 'vascular_lesion'
        }
        
        self.data_df['diagnosis'] = self.data_df['diagnosis'].replace(diagnosis_mapping)
        
        # إضافة امتداد الصورة إذا لم يكن موجوداً
        if not self.data_df['image_id'].str.contains('.').any():
            self.data_df['image_id'] = self.data_df['image_id'] + '.jpg'
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """إنشاء خريطة الفئات"""
        unique_classes = sorted(self.data_df['diagnosis'].unique())
        return {cls: idx for idx, cls in enumerate(unique_classes)}
    
    def setup(self, stage: Optional[str] = None):
        """إعداد مجموعات البيانات"""
        if stage == 'fit' or stage is None:
            # تقسيم البيانات
            train_df, temp_df = train_test_split(
                self.data_df,
                test_size=1 - self.config['data']['train_split'],
                stratify=self.data_df['diagnosis'],
                random_state=42
            )
            
            val_size = self.config['data']['val_split'] / (
                self.config['data']['val_split'] + self.config['data']['test_split']
            )
            
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size,
                stratify=temp_df['diagnosis'],
                random_state=42
            )
            
            # إنشاء التحويلات
            train_transform = self._get_train_transforms()
            val_transform = self._get_val_transforms()
            
            # إنشاء مجموعات البيانات
            self.train_dataset = SkinDiseaseDataset(
                train_df,
                self.data_path / 'images',
                train_transform,
                self.class_to_idx
            )
            
            self.val_dataset = SkinDiseaseDataset(
                val_df,
                self.data_path / 'images',
                val_transform,
                self.class_to_idx
            )
            
            self.test_dataset = SkinDiseaseDataset(
                test_df,
                self.data_path / 'images',
                val_transform,
                self.class_to_idx
            )
            
            # حساب أوزان الفئات
            self.class_weights = self.train_dataset.get_class_weights()
            
            print(f"التدريب: {len(self.train_dataset)} عينة")
            print(f"التحقق: {len(self.val_dataset)} عينة")
            print(f"الاختبار: {len(self.test_dataset)} عينة")
    
    def _get_train_transforms(self) -> A.Compose:
        """تحويلات التدريب مع تحسين البيانات"""
        aug_config = self.config['augmentation']
        
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=aug_config['horizontal_flip']),
            A.VerticalFlip(p=aug_config['vertical_flip']),
            A.Rotate(limit=aug_config['rotation_degrees'], p=0.5),
            A.ColorJitter(
                brightness=aug_config['brightness'],
                contrast=aug_config['contrast'],
                saturation=aug_config['saturation'],
                hue=aug_config['hue'],
                p=0.5
            ),
            A.GaussianBlur(blur_limit=3, p=aug_config['gaussian_blur']),
            A.ElasticTransform(p=aug_config['elastic_transform']),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=aug_config['cutout']
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _get_val_transforms(self) -> A.Compose:
        """تحويلات التحقق والاختبار"""
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def train_dataloader(self) -> DataLoader:
        """محمل بيانات التدريب"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """محمل بيانات التحقق"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """محمل بيانات الاختبار"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

def create_sample_dataset(output_dir: Union[str, Path], num_samples: int = 1000):
    """
    إنشاء مجموعة بيانات تجريبية للاختبار
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # إنشاء مجلد الصور
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # فئات الأمراض
    diseases = [
        'normal', 'melanoma', 'psoriasis', 'eczema', 'contact_dermatitis',
        'rosacea', 'acne', 'warts', 'fungal_infection', 'vitiligo'
    ]
    
    # إنشاء البيانات التجريبية
    data = []
    for i in range(num_samples):
        image_id = f'sample_{i:04d}.jpg'
        diagnosis = np.random.choice(diseases)
        
        # إنشاء صورة تجريبية
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_pil = Image.fromarray(image)
        image_pil.save(images_dir / image_id)
        
        data.append({
            'image_id': image_id,
            'diagnosis': diagnosis,
            'age': np.random.randint(18, 80),
            'sex': np.random.choice(['male', 'female']),
            'localization': np.random.choice(['face', 'arm', 'leg', 'torso'])
        })
    
    # حفظ ملف CSV
    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    print(f"تم إنشاء مجموعة بيانات تجريبية في {output_dir}")
    print(f"عدد العينات: {num_samples}")
    print(f"توزيع الفئات:")
    for disease, count in df['diagnosis'].value_counts().items():
        print(f"  {disease}: {count}")

if __name__ == "__main__":
    # إنشاء مجموعة بيانات تجريبية
    create_sample_dataset('data/sample_dataset', num_samples=1000)
    
    # اختبار محمل البيانات
    from config import get_config
    
    config = get_config()
    config['data']['data_path'] = Path('data/sample_dataset')
    
    data_module = SkinDiseaseDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    
    # اختبار التحميل
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    
    print(f"شكل الدفعة: {images.shape}")
    print(f"التسميات: {labels}")
    print(f"نطاق القيم: [{images.min():.3f}, {images.max():.3f}]")