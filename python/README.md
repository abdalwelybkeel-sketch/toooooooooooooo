# نموذج تحليل أمراض الجلد - PyTorch

هذا المجلد يحتوي على جميع أكواد التدريب والتطوير لنموذج الذكاء الاصطناعي لتحليل أمراض الجلد باستخدام PyTorch.

## هيكل المجلد

```
python/
├── data/                    # مجلد البيانات
│   ├── raw/                # البيانات الخام
│   ├── processed/          # البيانات المعالجة
│   └── splits/             # تقسيم البيانات (train/val/test)
├── models/                 # نماذج PyTorch
│   ├── cnn_model.py       # نموذج CNN مخصص
│   ├── resnet_model.py    # نموذج ResNet
│   └── efficientnet_model.py # نموذج EfficientNet
├── utils/                  # أدوات مساعدة
│   ├── data_loader.py     # تحميل البيانات
│   ├── transforms.py      # تحويلات الصور
│   ├── metrics.py         # مقاييس الأداء
│   └── visualization.py   # رسم النتائج
├── training/               # سكريبتات التدريب
│   ├── train.py           # التدريب الرئيسي
│   ├── validate.py        # التحقق من النموذج
│   └── config.py          # إعدادات التدريب
├── inference/              # الاستنتاج والتصدير
│   ├── predict.py         # التنبؤ
│   ├── export_model.py    # تصدير النموذج
│   └── test_model.py      # اختبار النموذج
├── notebooks/              # دفاتر Jupyter
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── requirements.txt        # المتطلبات
├── setup.py               # إعداد المشروع
└── README.md              # هذا الملف
```

## المتطلبات

```bash
pip install -r requirements.txt
```

## البيانات المدعومة

- **ISIC Dataset**: مجموعة بيانات الجمعية الدولية لتصوير الجلد
- **HAM10000**: مجموعة بيانات الآفات الجلدية الصبغية
- **DermNet**: مجموعة بيانات الأمراض الجلدية
- **Custom Dataset**: بيانات مخصصة

## الأمراض المدعومة

1. **طبيعي** - Normal Skin
2. **سرطان الجلد** - Melanoma
3. **الصدفية** - Psoriasis
4. **الأكزيما** - Eczema
5. **التهاب الجلد التماسي** - Contact Dermatitis
6. **الوردية** - Rosacea
7. **حب الشباب** - Acne
8. **الثآليل** - Warts
9. **الفطريات الجلدية** - Fungal Infections
10. **البهاق** - Vitiligo

## كيفية الاستخدام

### 1. تحضير البيانات

```bash
python utils/data_loader.py --dataset_path /path/to/dataset --output_path data/processed/
```

### 2. التدريب

```bash
python training/train.py --config training/config.py --model resnet50 --epochs 100
```

### 3. التحقق من النموذج

```bash
python training/validate.py --model_path models/best_model.pth --test_data data/splits/test/
```

### 4. تصدير النموذج للإنتاج

```bash
python inference/export_model.py --model_path models/best_model.pth --output_path ../lib/assets/models/skin_disease_model.pt
```

### 5. اختبار النموذج

```bash
python inference/test_model.py --model_path models/best_model.pth --image_path test_image.jpg
```

## النماذج المدعومة

### 1. CNN مخصص
- نموذج شبكة عصبية تطبيقية مصمم خصيصاً لتحليل أمراض الجلد
- سريع ومناسب للأجهزة المحمولة

### 2. ResNet
- نماذج ResNet18, ResNet34, ResNet50, ResNet101
- دقة عالية مع Transfer Learning

### 3. EfficientNet
- نماذج EfficientNet-B0 إلى B7
- توازن ممتاز بين الدقة والسرعة

## مقاييس الأداء

- **Accuracy**: دقة التصنيف العامة
- **Precision**: دقة كل فئة
- **Recall**: استدعاء كل فئة
- **F1-Score**: المتوسط التوافقي للدقة والاستدعاء
- **AUC-ROC**: منطقة تحت منحنى ROC
- **Confusion Matrix**: مصفوفة الخلط

## التحسينات المطبقة

### 1. تحسين البيانات (Data Augmentation)
- التدوير والانعكاس
- تغيير السطوع والتباين
- القص العشوائي
- التشويه المرن

### 2. تقنيات التدريب
- Learning Rate Scheduling
- Early Stopping
- Gradient Clipping
- Mixed Precision Training

### 3. تحسين النموذج
- Dropout للتنظيم
- Batch Normalization
- Label Smoothing
- Test Time Augmentation

## النتائج المتوقعة

| النموذج | الدقة | الحجم | السرعة |
|---------|-------|-------|---------|
| CNN مخصص | 85% | 15MB | سريع |
| ResNet50 | 92% | 98MB | متوسط |
| EfficientNet-B3 | 94% | 48MB | متوسط |

## التكامل مع Flutter

النموذج المدرب يتم تصديره بصيغة `.pt` ليتم استخدامه في تطبيق Flutter عبر مكتبة `pytorch_lite`.

## المساهمة

1. Fork المشروع
2. إنشاء فرع للميزة الجديدة
3. Commit التغييرات
4. Push للفرع
5. إنشاء Pull Request

## الترخيص

هذا المشروع مرخص تحت رخصة MIT - انظر ملف LICENSE للتفاصيل.

## الدعم

للحصول على الدعم أو الإبلاغ عن مشاكل، يرجى إنشاء Issue في المستودع.