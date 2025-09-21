"""
تصدير النموذج المدرب للاستخدام في تطبيق Flutter
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.jit import script, trace
import onnx
import onnxruntime as ort

# إضافة مسار المشروع
sys.path.append(str(Path(__file__).parent.parent))

from config import Config, get_config
from models.efficientnet_model import create_efficientnet_model
from training.train import SkinDiseaseClassifier

warnings.filterwarnings("ignore")

class ModelExporter:
    """
    فئة تصدير النماذج لصيغ مختلفة
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.export_config = config['export']
        self.output_dir = Path(self.export_config['output_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_trained_model(self, checkpoint_path: str) -> nn.Module:
        """
        تحميل النموذج المدرب من checkpoint
        """
        print(f"تحميل النموذج من: {checkpoint_path}")
        
        # تحميل checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # إنشاء النموذج
        model = SkinDiseaseClassifier(self.config)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print("تم تحميل النموذج بنجاح")
        return model
    
    def export_to_torchscript(
        self, 
        model: nn.Module, 
        input_shape: tuple = (1, 3, 224, 224)
    ) -> str:
        """
        تصدير النموذج إلى TorchScript
        """
        print("تصدير النموذج إلى TorchScript...")
        
        # إنشاء مثال للإدخال
        example_input = torch.randn(input_shape)
        
        try:
            # محاولة استخدام torch.jit.trace
            traced_model = trace(model, example_input)
            
            # تحسين النموذج للأجهزة المحمولة
            if self.export_config['optimize_for_mobile']:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                traced_model = optimize_for_mobile(traced_model)
            
            # حفظ النموذج
            output_path = self.output_dir / self.export_config['model_name']
            traced_model.save(str(output_path))
            
            print(f"تم تصدير النموذج إلى: {output_path}")
            
            # اختبار النموذج المصدر
            self._test_torchscript_model(str(output_path), input_shape)
            
            return str(output_path)
            
        except Exception as e:
            print(f"فشل في تصدير TorchScript: {e}")
            
            # محاولة استخدام torch.jit.script
            try:
                scripted_model = script(model)
                output_path = self.output_dir / self.export_config['model_name']
                scripted_model.save(str(output_path))
                
                print(f"تم تصدير النموذج باستخدام script إلى: {output_path}")
                return str(output_path)
                
            except Exception as e2:
                print(f"فشل في تصدير script أيضاً: {e2}")
                raise e2
    
    def export_to_onnx(
        self, 
        model: nn.Module, 
        input_shape: tuple = (1, 3, 224, 224)
    ) -> str:
        """
        تصدير النموذج إلى ONNX
        """
        print("تصدير النموذج إلى ONNX...")
        
        # إنشاء مثال للإدخال
        example_input = torch.randn(input_shape)
        
        # مسار الإخراج
        output_path = self.output_dir / (self.export_config['model_name'].replace('.pt', '.onnx'))
        
        try:
            torch.onnx.export(
                model,
                example_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"تم تصدير النموذج إلى: {output_path}")
            
            # التحقق من النموذج
            self._verify_onnx_model(str(output_path))
            
            # اختبار النموذج
            self._test_onnx_model(str(output_path), input_shape)
            
            return str(output_path)
            
        except Exception as e:
            print(f"فشل في تصدير ONNX: {e}")
            raise e
    
    def _test_torchscript_model(self, model_path: str, input_shape: tuple):
        """
        اختبار النموذج المصدر بصيغة TorchScript
        """
        print("اختبار النموذج المصدر...")
        
        try:
            # تحميل النموذج
            loaded_model = torch.jit.load(model_path)
            loaded_model.eval()
            
            # إنشاء إدخال تجريبي
            test_input = torch.randn(input_shape)
            
            # التنبؤ
            with torch.no_grad():
                output = loaded_model(test_input)
            
            print(f"شكل الإخراج: {output.shape}")
            print(f"نطاق الإخراج: [{output.min():.4f}, {output.max():.4f}]")
            print("اختبار النموذج نجح ✓")
            
        except Exception as e:
            print(f"فشل في اختبار النموذج: {e}")
            raise e
    
    def _verify_onnx_model(self, model_path: str):
        """
        التحقق من صحة نموذج ONNX
        """
        try:
            # تحميل النموذج
            onnx_model = onnx.load(model_path)
            
            # التحقق من النموذج
            onnx.checker.check_model(onnx_model)
            
            print("التحقق من نموذج ONNX نجح ✓")
            
        except Exception as e:
            print(f"فشل في التحقق من نموذج ONNX: {e}")
            raise e
    
    def _test_onnx_model(self, model_path: str, input_shape: tuple):
        """
        اختبار نموذج ONNX
        """
        print("اختبار نموذج ONNX...")
        
        try:
            # إنشاء جلسة ONNX Runtime
            session = ort.InferenceSession(model_path)
            
            # إنشاء إدخال تجريبي
            test_input = torch.randn(input_shape).numpy()
            
            # التنبؤ
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})
            
            print(f"شكل الإخراج: {output[0].shape}")
            print(f"نطاق الإخراج: [{output[0].min():.4f}, {output[0].max():.4f}]")
            print("اختبار نموذج ONNX نجح ✓")
            
        except Exception as e:
            print(f"فشل في اختبار نموذج ONNX: {e}")
            raise e
    
    def quantize_model(self, model_path: str) -> str:
        """
        ضغط النموذج باستخدام Quantization
        """
        if not self.export_config['quantization']:
            return model_path
        
        print("ضغط النموذج...")
        
        try:
            # تحميل النموذج
            model = torch.jit.load(model_path)
            
            # تطبيق quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            # حفظ النموذج المضغوط
            quantized_path = model_path.replace('.pt', '_quantized.pt')
            quantized_model.save(quantized_path)
            
            # مقارنة الأحجام
            original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
            
            print(f"الحجم الأصلي: {original_size:.2f} MB")
            print(f"الحجم المضغوط: {quantized_size:.2f} MB")
            print(f"نسبة الضغط: {(1 - quantized_size/original_size)*100:.1f}%")
            
            return quantized_path
            
        except Exception as e:
            print(f"فشل في ضغط النموذج: {e}")
            return model_path
    
    def create_model_info(self, model_path: str, original_checkpoint: str):
        """
        إنشاء ملف معلومات النموذج
        """
        print("إنشاء ملف معلومات النموذج...")
        
        # معلومات النموذج
        model_info = {
            'model_name': self.export_config['model_name'],
            'architecture': self.config['model']['architecture'],
            'num_classes': self.config['num_classes'],
            'input_size': self.config['data']['image_size'],
            'classes': self.config['classes'],
            'class_names_ar': self.config['class_names_ar'],
            'export_format': self.export_config['export_format'],
            'optimized_for_mobile': self.export_config['optimize_for_mobile'],
            'quantized': self.export_config['quantization'],
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'original_checkpoint': original_checkpoint,
            'export_date': str(torch.datetime.datetime.now()),
        }
        
        # حفظ معلومات النموذج
        import json
        info_path = self.output_dir / 'model_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print(f"تم حفظ معلومات النموذج في: {info_path}")
        
        return model_info

def main():
    """
    الدالة الرئيسية لتصدير النموذج
    """
    parser = argparse.ArgumentParser(description='تصدير نموذج تحليل أمراض الجلد')
    parser.add_argument('--checkpoint', type=str, required=True, help='مسار checkpoint النموذج')
    parser.add_argument('--format', type=str, default='torchscript', 
                       choices=['torchscript', 'onnx', 'both'], help='صيغة التصدير')
    parser.add_argument('--output_dir', type=str, help='مجلد الإخراج')
    parser.add_argument('--model_name', type=str, help='اسم النموذج المصدر')
    parser.add_argument('--optimize_mobile', action='store_true', help='تحسين للأجهزة المحمولة')
    parser.add_argument('--quantize', action='store_true', help='ضغط النموذج')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224], help='حجم الإدخال')
    
    args = parser.parse_args()
    
    # التحقق من وجود checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"لم يتم العثور على checkpoint: {args.checkpoint}")
    
    # تحميل الإعدادات
    config = get_config()
    
    # تحديث الإعدادات من المعاملات
    if args.output_dir:
        config['export']['output_path'] = Path(args.output_dir)
    if args.model_name:
        config['export']['model_name'] = args.model_name
    if args.optimize_mobile:
        config['export']['optimize_for_mobile'] = True
    if args.quantize:
        config['export']['quantization'] = True
    
    # تحديث حجم الإدخال
    config['data']['image_size'] = tuple(args.input_size)
    
    print("بدء تصدير النموذج...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"صيغة التصدير: {args.format}")
    print(f"مجلد الإخراج: {config['export']['output_path']}")
    print(f"حجم الإدخال: {config['data']['image_size']}")
    
    # إنشاء مصدر النموذج
    exporter = ModelExporter(config)
    
    # تحميل النموذج المدرب
    model = exporter.load_trained_model(args.checkpoint)
    
    # شكل الإدخال
    input_shape = (1, 3, *config['data']['image_size'])
    
    exported_paths = []
    
    # تصدير إلى TorchScript
    if args.format in ['torchscript', 'both']:
        try:
            torchscript_path = exporter.export_to_torchscript(model, input_shape)
            
            # ضغط النموذج إذا طُلب ذلك
            if config['export']['quantization']:
                torchscript_path = exporter.quantize_model(torchscript_path)
            
            exported_paths.append(torchscript_path)
            
        except Exception as e:
            print(f"فشل في تصدير TorchScript: {e}")
    
    # تصدير إلى ONNX
    if args.format in ['onnx', 'both']:
        try:
            onnx_path = exporter.export_to_onnx(model, input_shape)
            exported_paths.append(onnx_path)
            
        except Exception as e:
            print(f"فشل في تصدير ONNX: {e}")
    
    # إنشاء ملف معلومات النموذج
    if exported_paths:
        model_info = exporter.create_model_info(exported_paths[0], args.checkpoint)
        
        print("\n" + "="*50)
        print("تم تصدير النموذج بنجاح!")
        print("="*50)
        print(f"النماذج المصدرة:")
        for path in exported_paths:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {path} ({size_mb:.2f} MB)")
        
        print(f"\nمعلومات النموذج:")
        print(f"  - البنية: {model_info['architecture']}")
        print(f"  - عدد الفئات: {model_info['num_classes']}")
        print(f"  - حجم الإدخال: {model_info['input_size']}")
        print(f"  - محسن للأجهزة المحمولة: {model_info['optimized_for_mobile']}")
        print(f"  - مضغوط: {model_info['quantized']}")
        print("="*50)
        
    else:
        print("فشل في تصدير النموذج!")
        sys.exit(1)

if __name__ == "__main__":
    main()