"""
إعدادات التدريب لنموذج تحليل أمراض الجلد
"""

import os
from pathlib import Path

# مسارات المشروع
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
LOGS_ROOT = PROJECT_ROOT / "logs"

# إنشاء المجلدات إذا لم تكن موجودة
for path in [DATA_ROOT, MODELS_ROOT, LOGS_ROOT]:
    path.mkdir(exist_ok=True)

class Config:
    """إعدادات التدريب الرئيسية"""
    
    # إعدادات البيانات
    DATA_CONFIG = {
        'dataset_name': 'skin_diseases',
        'data_path': DATA_ROOT / 'processed',
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'image_size': (224, 224),
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True,
    }
    
    # فئات الأمراض
    DISEASE_CLASSES = [
        'normal',           # طبيعي
        'melanoma',         # سرطان الجلد
        'psoriasis',        # الصدفية
        'eczema',           # الأكزيما
        'contact_dermatitis', # التهاب الجلد التماسي
        'rosacea',          # الوردية
        'acne',             # حب الشباب
        'warts',            # الثآليل
        'fungal_infection', # الفطريات الجلدية
        'vitiligo',         # البهاق
    ]
    
    # أسماء الأمراض بالعربية
    DISEASE_NAMES_AR = {
        'normal': 'طبيعي',
        'melanoma': 'سرطان الجلد',
        'psoriasis': 'الصدفية',
        'eczema': 'الأكزيما',
        'contact_dermatitis': 'التهاب الجلد التماسي',
        'rosacea': 'الوردية',
        'acne': 'حب الشباب',
        'warts': 'الثآليل',
        'fungal_infection': 'الفطريات الجلدية',
        'vitiligo': 'البهاق',
    }
    
    NUM_CLASSES = len(DISEASE_CLASSES)
    
    # إعدادات النموذج
    MODEL_CONFIG = {
        'architecture': 'efficientnet_b3',  # resnet50, efficientnet_b3, custom_cnn
        'pretrained': True,
        'dropout_rate': 0.3,
        'use_attention': True,
        'freeze_backbone': False,
    }
    
    # إعدادات التدريب
    TRAINING_CONFIG = {
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',  # adam, adamw, sgd
        'scheduler': 'cosine',  # cosine, step, plateau
        'warmup_epochs': 5,
        'gradient_clip_norm': 1.0,
        'mixed_precision': True,
        'early_stopping_patience': 15,
        'save_top_k': 3,
    }
    
    # إعدادات تحسين البيانات
    AUGMENTATION_CONFIG = {
        'horizontal_flip': 0.5,
        'vertical_flip': 0.3,
        'rotation_degrees': 30,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
        'gaussian_blur': 0.1,
        'elastic_transform': 0.2,
        'cutout': 0.3,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
    }
    
    # إعدادات التحقق
    VALIDATION_CONFIG = {
        'val_check_interval': 0.25,  # التحقق كل ربع epoch
        'log_every_n_steps': 50,
        'save_predictions': True,
        'compute_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    }
    
    # إعدادات الحفظ والتسجيل
    LOGGING_CONFIG = {
        'experiment_name': 'skin_disease_classification',
        'log_dir': LOGS_ROOT,
        'save_dir': MODELS_ROOT,
        'use_wandb': True,
        'wandb_project': 'skin-disease-analyzer',
        'save_hyperparameters': True,
        'log_model': True,
    }
    
    # إعدادات الأجهزة
    HARDWARE_CONFIG = {
        'accelerator': 'auto',  # auto, gpu, cpu
        'devices': 'auto',
        'precision': 16,  # 16, 32
        'strategy': 'auto',
        'sync_batchnorm': True,
    }
    
    # إعدادات التصدير
    EXPORT_CONFIG = {
        'export_format': 'torchscript',  # torchscript, onnx
        'optimize_for_mobile': True,
        'quantization': False,
        'output_path': PROJECT_ROOT.parent / 'lib' / 'assets' / 'models',
        'model_name': 'skin_disease_model.pt',
    }
    
    # إعدادات التقييم
    EVALUATION_CONFIG = {
        'test_time_augmentation': True,
        'tta_transforms': 5,
        'confidence_threshold': 0.5,
        'generate_reports': True,
        'save_confusion_matrix': True,
        'save_classification_report': True,
    }

class DatasetConfig:
    """إعدادات مجموعات البيانات المختلفة"""
    
    # ISIC Dataset
    ISIC_CONFIG = {
        'name': 'ISIC',
        'url': 'https://challenge.isic-archive.com/data/',
        'classes': ['melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis', 
                   'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma'],
        'image_format': 'jpg',
        'metadata_format': 'csv',
    }
    
    # HAM10000 Dataset
    HAM10000_CONFIG = {
        'name': 'HAM10000',
        'url': 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T',
        'classes': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
        'image_format': 'jpg',
        'metadata_format': 'csv',
    }
    
    # DermNet Dataset
    DERMNET_CONFIG = {
        'name': 'DermNet',
        'url': 'http://www.dermnet.com/',
        'classes': ['acne', 'eczema', 'psoriasis', 'rosacea', 'vitiligo', 'warts'],
        'image_format': 'jpg',
        'metadata_format': 'folder_structure',
    }

class ModelArchitectures:
    """تكوينات النماذج المختلفة"""
    
    RESNET_CONFIGS = {
        'resnet18': {'layers': [2, 2, 2, 2], 'params': '11.7M'},
        'resnet34': {'layers': [3, 4, 6, 3], 'params': '21.8M'},
        'resnet50': {'layers': [3, 4, 6, 3], 'params': '25.6M'},
        'resnet101': {'layers': [3, 4, 23, 3], 'params': '44.5M'},
    }
    
    EFFICIENTNET_CONFIGS = {
        'efficientnet_b0': {'width': 1.0, 'depth': 1.0, 'resolution': 224, 'params': '5.3M'},
        'efficientnet_b1': {'width': 1.0, 'depth': 1.1, 'resolution': 240, 'params': '7.8M'},
        'efficientnet_b2': {'width': 1.1, 'depth': 1.2, 'resolution': 260, 'params': '9.2M'},
        'efficientnet_b3': {'width': 1.2, 'depth': 1.4, 'resolution': 300, 'params': '12M'},
        'efficientnet_b4': {'width': 1.4, 'depth': 1.8, 'resolution': 380, 'params': '19M'},
    }
    
    CUSTOM_CNN_CONFIG = {
        'name': 'CustomCNN',
        'layers': [
            {'type': 'conv', 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'conv', 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'conv', 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'conv', 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'fc', 'out_features': 512},
            {'type': 'fc', 'out_features': 256},
        ],
        'params': '~15M'
    }

# إعدادات البيئة
ENVIRONMENT = {
    'CUDA_VISIBLE_DEVICES': '0',
    'PYTHONPATH': str(PROJECT_ROOT),
    'WANDB_API_KEY': os.getenv('WANDB_API_KEY', ''),
    'HF_TOKEN': os.getenv('HF_TOKEN', ''),
}

# إعدادات التطوير
DEBUG_CONFIG = {
    'fast_dev_run': False,
    'overfit_batches': 0,
    'limit_train_batches': 1.0,
    'limit_val_batches': 1.0,
    'limit_test_batches': 1.0,
    'profiler': None,  # 'simple', 'advanced', 'pytorch'
}

def get_config():
    """إرجاع إعدادات التدريب الكاملة"""
    return {
        'data': Config.DATA_CONFIG,
        'model': Config.MODEL_CONFIG,
        'training': Config.TRAINING_CONFIG,
        'augmentation': Config.AUGMENTATION_CONFIG,
        'validation': Config.VALIDATION_CONFIG,
        'logging': Config.LOGGING_CONFIG,
        'hardware': Config.HARDWARE_CONFIG,
        'export': Config.EXPORT_CONFIG,
        'evaluation': Config.EVALUATION_CONFIG,
        'classes': Config.DISEASE_CLASSES,
        'class_names_ar': Config.DISEASE_NAMES_AR,
        'num_classes': Config.NUM_CLASSES,
    }

def print_config():
    """طباعة الإعدادات الحالية"""
    config = get_config()
    print("=" * 50)
    print("إعدادات التدريب")
    print("=" * 50)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {settings}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_config()