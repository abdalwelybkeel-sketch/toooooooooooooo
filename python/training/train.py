"""
سكريبت التدريب الرئيسي لنموذج تحليل أمراض الجلد
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# إضافة مسار المشروع
sys.path.append(str(Path(__file__).parent.parent))

from config import Config, get_config
from models.efficientnet_model import create_efficientnet_model
from utils.data_loader import SkinDiseaseDataModule
from utils.metrics import compute_metrics
from utils.visualization import plot_training_curves

warnings.filterwarnings("ignore")

class SkinDiseaseClassifier(pl.LightningModule):
    """
    نموذج تصنيف أمراض الجلد باستخدام PyTorch Lightning
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config)
        
        # إنشاء النموذج
        self.model = create_efficientnet_model(
            model_name=config['model']['architecture'],
            num_classes=config['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout_rate=config['model']['dropout_rate'],
            use_attention=config['model']['use_attention'],
        )
        
        # دالة الخسارة
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1  # Label smoothing للتنظيم
        )
        
        # متغيرات لحفظ النتائج
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # أفضل دقة
        self.best_val_acc = 0.0
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """خطوة التدريب"""
        images, labels = batch
        
        # التمرير الأمامي
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # حساب الدقة
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # تسجيل المقاييس
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # حفظ النتائج
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach(),
            'preds': preds.detach(),
            'labels': labels.detach()
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """خطوة التحقق"""
        images, labels = batch
        
        # التمرير الأمامي
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # حساب الدقة
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # تسجيل المقاييس
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        # حفظ النتائج
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach(),
            'preds': preds.detach(),
            'labels': labels.detach()
        })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """خطوة الاختبار"""
        images, labels = batch
        
        # التمرير الأمامي
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # حساب الدقة
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # تسجيل المقاييس
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        # حفظ النتائج
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach(),
            'preds': preds.detach(),
            'labels': labels.detach()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        """نهاية epoch التدريب"""
        if not self.training_step_outputs:
            return
        
        # حساب المتوسطات
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()
        
        # تسجيل المقاييس
        self.log('train_epoch_loss', avg_loss)
        self.log('train_epoch_acc', avg_acc)
        
        # مسح النتائج
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """نهاية epoch التحقق"""
        if not self.validation_step_outputs:
            return
        
        # حساب المتوسطات
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
        
        # جمع جميع التنبؤات والتسميات
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # حساب المقاييس التفصيلية
        metrics = compute_metrics(
            all_preds.cpu().numpy(),
            all_labels.cpu().numpy(),
            num_classes=self.config['num_classes']
        )
        
        # تسجيل المقاييس
        self.log('val_epoch_loss', avg_loss)
        self.log('val_epoch_acc', avg_acc)
        self.log('val_precision', metrics['precision'])
        self.log('val_recall', metrics['recall'])
        self.log('val_f1', metrics['f1'])
        
        # تحديث أفضل دقة
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            self.log('best_val_acc', self.best_val_acc)
        
        # مسح النتائج
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """نهاية epoch الاختبار"""
        if not self.test_step_outputs:
            return
        
        # حساب المتوسطات
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.test_step_outputs]).mean()
        
        # جمع جميع التنبؤات والتسميات
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # حساب المقاييس التفصيلية
        metrics = compute_metrics(
            all_preds.cpu().numpy(),
            all_labels.cpu().numpy(),
            num_classes=self.config['num_classes']
        )
        
        # تسجيل المقاييس
        self.log('test_epoch_loss', avg_loss)
        self.log('test_epoch_acc', avg_acc)
        self.log('test_precision', metrics['precision'])
        self.log('test_recall', metrics['recall'])
        self.log('test_f1', metrics['f1'])
        
        # طباعة النتائج النهائية
        print(f"\n{'='*50}")
        print("نتائج الاختبار النهائية:")
        print(f"{'='*50}")
        print(f"الدقة: {avg_acc:.4f}")
        print(f"الدقة (Precision): {metrics['precision']:.4f}")
        print(f"الاستدعاء (Recall): {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"{'='*50}")
        
        # مسح النتائج
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """إعداد المحسن والجدولة"""
        
        # اختيار المحسن
        if self.config['training']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        
        # اختيار الجدولة
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['epochs']
            )
        elif self.config['training']['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss' if self.config['training']['scheduler'] == 'plateau' else None,
            }
        }

def setup_callbacks(config: Dict[str, Any]) -> list:
    """إعداد callbacks للتدريب"""
    
    callbacks = []
    
    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['save_dir'],
        filename='{epoch}-{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        save_top_k=config['training']['save_top_k'],
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Progress Bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks

def setup_logger(config: Dict[str, Any]) -> Optional[pl.loggers.Logger]:
    """إعداد logger للتدريب"""
    
    if config['logging']['use_wandb']:
        try:
            logger = WandbLogger(
                project=config['logging']['wandb_project'],
                name=config['logging']['experiment_name'],
                save_dir=config['logging']['log_dir']
            )
            return logger
        except Exception as e:
            print(f"فشل في إعداد Wandb: {e}")
            print("سيتم استخدام TensorBoard بدلاً من ذلك")
    
    # استخدام TensorBoard كبديل
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=config['logging']['experiment_name']
    )
    
    return logger

def main():
    """الدالة الرئيسية للتدريب"""
    
    # تحليل المعاملات
    parser = argparse.ArgumentParser(description='تدريب نموذج تحليل أمراض الجلد')
    parser.add_argument('--config', type=str, help='مسار ملف الإعدادات')
    parser.add_argument('--model', type=str, default='efficientnet_b3', help='نوع النموذج')
    parser.add_argument('--epochs', type=int, default=100, help='عدد epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='حجم الدفعة')
    parser.add_argument('--lr', type=float, default=1e-4, help='معدل التعلم')
    parser.add_argument('--data_path', type=str, help='مسار البيانات')
    parser.add_argument('--resume', type=str, help='مسار checkpoint للاستكمال')
    parser.add_argument('--fast_dev_run', action='store_true', help='تشغيل سريع للتطوير')
    
    args = parser.parse_args()
    
    # تحميل الإعدادات
    config = get_config()
    
    # تحديث الإعدادات من المعاملات
    if args.model:
        config['model']['architecture'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.data_path:
        config['data']['data_path'] = Path(args.data_path)
    
    # طباعة الإعدادات
    print("بدء التدريب بالإعدادات التالية:")
    print(f"النموذج: {config['model']['architecture']}")
    print(f"عدد epochs: {config['training']['epochs']}")
    print(f"حجم الدفعة: {config['data']['batch_size']}")
    print(f"معدل التعلم: {config['training']['learning_rate']}")
    print(f"عدد الفئات: {config['num_classes']}")
    
    # إعداد البيانات
    print("تحميل البيانات...")
    data_module = SkinDiseaseDataModule(config)
    data_module.setup()
    
    # إنشاء النموذج
    print("إنشاء النموذج...")
    model = SkinDiseaseClassifier(config)
    
    # إعداد callbacks و logger
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # إعداد المدرب
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config['training']['gradient_clip_norm'],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=config['validation']['log_every_n_steps'],
        val_check_interval=config['validation']['val_check_interval'],
    )
    
    # بدء التدريب
    print("بدء التدريب...")
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)
    
    # الاختبار النهائي
    print("بدء الاختبار النهائي...")
    trainer.test(model, data_module)
    
    # حفظ النموذج النهائي
    final_model_path = config['logging']['save_dir'] / 'final_model.ckpt'
    trainer.save_checkpoint(final_model_path)
    print(f"تم حفظ النموذج النهائي في: {final_model_path}")
    
    print("اكتمل التدريب بنجاح!")

if __name__ == "__main__":
    main()