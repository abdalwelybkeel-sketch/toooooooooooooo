"""
نموذج EfficientNet لتصنيف أمراض الجلد
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional, Dict, Any

class EfficientNetModel(nn.Module):
    """
    نموذج EfficientNet مع تخصيصات لتصنيف أمراض الجلد
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b3',
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        freeze_backbone: bool = False,
    ):
        super(EfficientNetModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # تحميل النموذج المدرب مسبقاً
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # إزالة طبقة التصنيف الأصلية
            global_pool='',  # إزالة Global Pooling
        )
        
        # الحصول على عدد الميزات
        self.feature_dim = self.backbone.num_features
        
        # تجميد الطبقات الأساسية إذا طُلب ذلك
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # طبقة الانتباه (Attention)
        if use_attention:
            self.attention = SpatialAttention(self.feature_dim)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # طبقات التصنيف
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # تهيئة الأوزان
        self._initialize_weights()
    
    def _initialize_weights(self):
        """تهيئة أوزان الطبقات الجديدة"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """التمرير الأمامي"""
        # استخراج الميزات من العمود الفقري
        features = self.backbone(x)
        
        # تطبيق الانتباه إذا كان مفعلاً
        if self.use_attention:
            features = self.attention(features)
        
        # Global Average Pooling
        features = self.global_pool(features)
        features = features.flatten(1)
        
        # التصنيف
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """استخراج الميزات فقط بدون التصنيف"""
        features = self.backbone(x)
        
        if self.use_attention:
            features = self.attention(features)
        
        features = self.global_pool(features)
        features = features.flatten(1)
        
        return features
    
    def unfreeze_backbone(self):
        """إلغاء تجميد العمود الفقري"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """تجميد العمود الفقري"""
        for param in self.backbone.parameters():
            param.requires_grad = False

class SpatialAttention(nn.Module):
    """
    طبقة الانتباه المكاني لتحسين التركيز على المناطق المهمة
    """
    
    def __init__(self, in_channels: int):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # حفظ الإدخال الأصلي
        residual = x
        
        # حساب خريطة الانتباه
        attention = self.conv1(x)
        attention = F.relu(attention, inplace=True)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # تطبيق الانتباه
        output = x * attention
        
        # إضافة الاتصال المتبقي
        output = output + residual
        
        return output

class EfficientNetWithAuxiliary(nn.Module):
    """
    نموذج EfficientNet مع مخرجات مساعدة لتحسين التدريب
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b3',
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        aux_weight: float = 0.3,
    ):
        super(EfficientNetWithAuxiliary, self).__init__()
        
        self.aux_weight = aux_weight
        
        # النموذج الرئيسي
        self.main_model = EfficientNetModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_attention=True,
        )
        
        # المصنف المساعد
        aux_feature_dim = self.main_model.feature_dim // 2
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(aux_feature_dim, num_classes)
        )
        
        # نقطة استخراج الميزات المساعدة
        self.aux_hook = None
        self._register_aux_hook()
    
    def _register_aux_hook(self):
        """تسجيل hook لاستخراج الميزات المساعدة"""
        def hook_fn(module, input, output):
            self.aux_features = output
        
        # تسجيل hook في منتصف النموذج تقريباً
        target_layer = list(self.main_model.backbone.children())[-3]
        self.aux_hook = target_layer.register_forward_hook(hook_fn)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """التمرير الأمامي مع المخرجات المساعدة"""
        # المخرج الرئيسي
        main_output = self.main_model(x)
        
        # المخرج المساعد
        aux_output = self.aux_classifier(self.aux_features)
        
        return {
            'main': main_output,
            'aux': aux_output
        }
    
    def __del__(self):
        """إزالة hook عند حذف الكائن"""
        if self.aux_hook is not None:
            self.aux_hook.remove()

def create_efficientnet_model(
    model_name: str = 'efficientnet_b3',
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    إنشاء نموذج EfficientNet
    
    Args:
        model_name: اسم النموذج (efficientnet_b0 إلى efficientnet_b7)
        num_classes: عدد الفئات
        pretrained: استخدام الأوزان المدربة مسبقاً
        **kwargs: معاملات إضافية
    
    Returns:
        النموذج المُنشأ
    """
    
    # التحقق من صحة اسم النموذج
    valid_models = [f'efficientnet_b{i}' for i in range(8)]
    if model_name not in valid_models:
        raise ValueError(f"Model {model_name} not supported. Choose from {valid_models}")
    
    # إنشاء النموذج
    model = EfficientNetModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
    
    return model

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    الحصول على معلومات النموذج
    
    Args:
        model_name: اسم النموذج
    
    Returns:
        معلومات النموذج
    """
    
    model_info = {
        'efficientnet_b0': {
            'params': '5.3M',
            'flops': '0.39B',
            'input_size': 224,
            'accuracy': '77.1%'
        },
        'efficientnet_b1': {
            'params': '7.8M',
            'flops': '0.70B',
            'input_size': 240,
            'accuracy': '79.1%'
        },
        'efficientnet_b2': {
            'params': '9.2M',
            'flops': '1.0B',
            'input_size': 260,
            'accuracy': '80.1%'
        },
        'efficientnet_b3': {
            'params': '12M',
            'flops': '1.8B',
            'input_size': 300,
            'accuracy': '81.6%'
        },
        'efficientnet_b4': {
            'params': '19M',
            'flops': '4.2B',
            'input_size': 380,
            'accuracy': '82.9%'
        },
        'efficientnet_b5': {
            'params': '30M',
            'flops': '9.9B',
            'input_size': 456,
            'accuracy': '83.6%'
        },
        'efficientnet_b6': {
            'params': '43M',
            'flops': '19B',
            'input_size': 528,
            'accuracy': '84.0%'
        },
        'efficientnet_b7': {
            'params': '66M',
            'flops': '37B',
            'input_size': 600,
            'accuracy': '84.3%'
        },
    }
    
    return model_info.get(model_name, {})

if __name__ == "__main__":
    # اختبار النموذج
    model = create_efficientnet_model(
        model_name='efficientnet_b3',
        num_classes=10,
        pretrained=True
    )
    
    # إنشاء tensor تجريبي
    x = torch.randn(2, 3, 300, 300)
    
    # التمرير الأمامي
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # معلومات النموذج
    info = get_model_info('efficientnet_b3')
    print(f"Model info: {info}")