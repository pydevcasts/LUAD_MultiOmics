# LUAD MultiOmics

این پروژه برای تحلیل و تشخیص سرطان آدنوکارسینوم ریه (LUAD) با رویکرد Multi-Omics و یادگیری ماشین طراحی شده است. هدف اصلی، آماده‌سازی داده‌ها، انتخاب ویژگی، آموزش مدل‌ها و اعتبارسنجی نهایی در یک پایپ‌لاین قابل‌اعتماد است.

## 1) خلاصهٔ وضعیت فعلی پروژه

نتایج فعلی پروژه بر اساس خروجی‌های ذخیره‌شده در پوشه `artifacts/` عبارت‌اند از:

| آزمایش | نتیجه | توضیح |
|---|---:|---|
| mRNA-only | BA = 1.0000، AUC = 1.0000 | تشخیص تومور در برابر نرمال با دقت کامل |
| miRNA-only | BA = 0.7505 | عملکرد خوب اما کمتر از mRNA |
| Clinical-only | BA = 0.6061 | عملکرد ضعیف‌تر نسبت به داده‌های مولکولی |
| Late Fusion (mRNA + miRNA + Clinical) | BA = 0.9490، AUC = 0.9970 | عملکرد بسیار قوی و پایدار |
| PSO Feature Selection | 20,531 → 324 ژن | کاهش شدید ابعاد بدون افت معنی‌دار در دقت |
| External Validation (GSE30219) | 307 نمونه تومور، mean probability = 0.9899 | اعتبارسنجی خارجی روی دادهٔ بیرونی |

نکته مهم: خروجی‌های بالا از فایل‌های JSON ذخیره‌شده در `artifacts/experiments/...` استخراج شده‌اند و به‌عنوان وضعیت فعلی پروژه محسوب می‌شوند.

## 2) نتایج ارزیابی به‌صورت خلاصه

### 2.1 ارزیابی درون‌نمونه‌ای

- mRNA-only: بهترین عملکرد، با BA و AUC برابر 1.0000
- miRNA-only: BA = 0.7505
- Late Fusion: BA = 0.9490، AUC = 0.9970
- Clinical-only: BA = 0.6061

### 2.2 اعتبارسنجی خارجی

- دیتاست بیرونی: GSE30219
- تعداد نمونه‌ها: 307
- نمونه‌های پیش‌بینی‌شده به‌عنوان Tumor: 307
- میانگین احتمال تومور: 0.9899
- برای این مجموعه، نمونه نرمال وجود ندارد و بنابراین ارزیابی نرمال/تومور کامل انجام نشده است.

## 3) نصب و آماده‌سازی محیط

### روش 1: Conda

```bash
conda env create -f environment.yml
conda activate luad_multiomics
```

### روش 2: Pip

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4) ساختار پروژه

```text
.
├── artifacts/               # خروجی‌های آزمایش‌ها و نتایج
├── configs/                 # تنظیمات پروژه و آزمایش‌ها
├── docs/                    # مستندات پروژه
├── scripts/                 # اسکریپت‌های اصلی اجرای پروژه
├── src/                     # کدهای اصلی پیاده‌سازی
├── environment.yml          # محیط Conda
├── requirements.txt         # وابستگی‌های Python
├── README.md                # این فایل
└── .gitignore
```

## 5) دستورات کامل اجرای پروژه

### 5.1 آماده‌سازی داده

```bash
python scripts/prepare_data.py
python scripts/build_cohort.py
python scripts/make_split.py
python scripts/prepare_clinical_features.py
python scripts/prepare_mrna_mirna_tn.py
python scripts/prepare_tumor_normal.py
python scripts/prepare_survival.py
python scripts/prepare_4modality_tn.py
```

### 5.2 آموزش مدل‌ها

```bash
python scripts/train_baseline.py
python scripts/train_late_fusion.py
python scripts/train_late_fusion_clinical.py
python scripts/train_late_fusion_pso.py
python scripts/train_pso_feature_selection.py
python scripts/train_pathway_survival.py
python scripts/train_survival.py
python scripts/train_tumor_normal.py
```

### 5.3 ارزیابی و اعتبارسنجی خارجی

```bash
python scripts/run_external_validation_gse30219.py
python scripts/validate_external_gse30219.py
python scripts/run_shap_analysis.py
python scripts/run_shap_biomarker_deep.py
```

### 5.4 اجرای کامل به‌ترتیب پیشنهادی

```bash
# 1. آماده‌سازی داده‌ها
python scripts/prepare_data.py
python scripts/build_cohort.py
python scripts/make_split.py
python scripts/prepare_clinical_features.py
python scripts/prepare_mrna_mirna_tn.py
python scripts/prepare_tumor_normal.py

# 2. آموزش مدل‌های پایه
python scripts/train_baseline.py
python scripts/train_pso_feature_selection.py
python scripts/train_late_fusion.py
python scripts/train_late_fusion_pso.py

# 3. تحلیل تفسیرپذیری
python scripts/run_shap_analysis.py
python scripts/run_shap_biomarker_deep.py

# 4. اعتبارسنجی بیرونی
python scripts/run_external_validation_gse30219.py
python scripts/validate_external_gse30219.py
```

## 6) تنظیم PYTHONPATH

اگر در اجرای اسکریپت‌ها با مشکل import روبرو شدید، مسیر کدها را به متغیر محیطی اضافه کنید:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

در PowerShell ویندوز:

```powershell
$env:PYTHONPATH += ";$PWD\src"
```

## 7) بررسی‌های عمومی پروژه

```bash
python -m compileall src
python -m compileall scripts
python -m pytest
```

## 8) نکات اجرایی مهم

- اسکریپت‌های اصلی پروژه باید از مسیر `scripts/` اجرا شوند.
- خروجی نتایج در پوشه `artifacts/` ذخیره می‌شود.
- برای گزارش و تحلیل‌های تفسیرپذیری، فایل‌های SHAP و نتایج Experimental در `artifacts/experiments/` قرار دارند.
- برای اجرای مدل‌های جدید، بهتر است ابتدا `configs/` و `datasets.yaml` را بررسی کنید و سپس دستورهای آموزش را اجرا کنید.

## 9) خلاصهٔ نتیجه‌گیری

در وضعیت فعلی، پروژه یک پایپ‌لاین معتبر برای طبقه‌بندی LUAD بر پایهٔ چند-اومیک دارد و نتایج ارزیابی نشان می‌دهد که Late Fusion و انتخاب ویژگی با PSO عملکرد بسیار خوبی دارند. برای افزایش اعتبار مقاله و ورود به مجلات سطح Q1/Q2، نیاز به چندین اقدام مهم در زمینه اعتبارسنجی خارجی، ارزیابی آماری، تکرارپذیری و گزارش‌سازی علمی وجود دارد.
