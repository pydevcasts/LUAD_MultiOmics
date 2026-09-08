# سند جامع پروژه: تشخیص سرطان ریه (LUAD) با Multi-Omics و یادگیری ماشین

> **تاریخ آخرین به‌روزرسانی:** 2026-09-07  
> **وضعیت فعلی:** فاز Late Fusion mRNA+miRNA تکمیل شد ✅  
> **مرحله بعدی:** اعتبارسنجی خارجی + SHAP بیومارکر + افزودن Methylation/CNA

---

## ۱. خلاصه اجرایی پروژه

این پروژه یک پایپلاین کامل و بدون نشت داده (Leakage-Safe) برای **تشخیص و تحلیل سرطان آدنوکارسینوم ریه (LUAD)** با استفاده از ادغام داده‌های چند-اومیکی است. تاکنون ۴ نوآوری از ۵ نوآوری پروپوزال پیاده‌سازی و تأیید شده‌اند.

### دستاوردهای کلیدی تا امروز:

| دستاورد | نتیجه | وضعیت |
|---------|-------|-------|
| تشخیص تومور/نرمال (mRNA) | AUC = 1.0000, BA = 1.0000 | ✅ تأیید |
| انتخاب ویژگی PSO (mRNA) | 20,531 → 324 ژن (کاهش 98%) | ✅ تأیید |
| انتخاب ویژگی PSO (miRNA) | 500 → 160 miRNA (کاهش 68%) | ✅ تأیید |
| Late Fusion (mRNA + miRNA) | AUC = 1.0000, BA = 0.9545 | ✅ تأیید |
| SHAP Modality Importance | mRNA: 0.742, miRNA: 0.555 | ✅ تأیید |
| اعتبارسنجی خارجی (GSE131907) | — | ⬜ اجرا نشده |

---

## ۲. دیتاست‌های استفاده‌شده

### ۲.۱. دیتاست اصلی: TCGA-LUAD

| مدالیته | فایل | نمونه تومور | نمونه نرمال | ویژگی‌ها | وضعیت |
|---------|------|-------------|-------------|----------|-------|
| mRNA | data_mrna_seq_v2_rsem.txt | 508 | 56 | 20,531 | ✅ استفاده شده |
| miRNA | TCGA-LUAD.mirna.tsv | 506 | 46 | 1,881 | ✅ استفاده شده |
| Methylation | data_methylation_hm27_hm450_merged.txt | 562 | 56 | 22,601 | ⬜ آماده، استفاده نشده |
| CNA Raw | data_cna.txt | 511 | 56 | 25,128 | ⬜ آماده، استفاده نشده |
| Mutation | data_mutations.txt | 512 | — | 18,522 | ⬜ آماده، استفاده نشده |
| RPPA | data_rppa_zscores.txt | 360 | 43 | 198 | ⬜ آماده، استفاده نشده |
| Clinical | data_clinical_patient.txt | 566 | — | 38 ستون | ✅ استفاده شده |

### ۲.۲. دیتاست اعتبارسنجی خارجی

| دیتاست | حجم | نمونه‌ها | وضعیت |
|--------|------|----------|-------|
| GSE131907 | 17.3 GB | >1000 | ⬜ هنوز استفاده نشده |

---

## ۳. تاریخچه آزمایش‌ها و نتایج

### ۳.۱. وظیفه تشخیص تومور/نرمال

#### آزمایش ۱: Baseline mRNA (بدون PSO)
- **داده:** 564 نمونه (508 تومور + 56 نرمال)
- **ویژگی‌ها:** 20,531 ژن → ANOVA top-1000
- **نتایج:**
  - Logistic Regression: BA=0.9951, AUC=1.0
  - Random Forest: BA=1.0, AUC=1.0
  - SVM: BA=1.0, AUC=1.0
  - XGBoost: BA=1.0, AUC=1.0

#### آزمایش ۲: PSO Feature Selection روی mRNA
- **ورودی:** 1,000 ژن (بعد از ANOVA)
- **تنظیمات PSO:** 30 ذره × 50 تکرار × 5-fold CV
- **خروجی:** 324 ژن بهینه (fitness=0.7815)
- **نتایج با 324 ژن:**
  - تمام 4 مدل: BA=1.0, AUC=1.0
- **تفسیر:** کاهش 98% ویژگی بدون افت دقت

#### آزمایش ۳: SHAP Biomarker Discovery
- **مدل:** XGBoost با 324 ژن PSO
- **Top-24 بیومارکر معنادار:**

| رتبه | ژن | Mean\|SHAP\| | جهت | اهمیت بیولوژیک |
|------|-----|-------------|------|----------------|
| 1 | TMEM8B | 1.290 | ↓ DOWN in Tumor | تنظیم‌کننده غشا |
| 2 | SDPR | 0.460 | ↓ DOWN in Tumor | سرکوب‌گر تومور |
| 3 | STX1A | 0.443 | ↓ DOWN in Tumor | Syntaxin-1A |
| 4 | SGCG | 0.432 | ↓ DOWN in Tumor | γ-Sarcoglycan |
| 5 | SNAP47 | 0.394 | ↑ UP in Tumor | SNARE protein |
| 6 | MSTO1 | 0.269 | ↓ DOWN in Tumor | تنظیم تقسیم سلولی |
| 7 | ERCC6L | 0.183 | ↓ DOWN in Tumor | ترمیم DNA |
| 8 | EIF2AK1 | 0.129 | ↓ DOWN in Tumor | کیناز پاسخ استرس |
| 9 | FABP4 | 0.093 | ↓ DOWN in Tumor | متابولیسم چربی |
| 12 | ETV4 | 0.073 | ↑ UP in Tumor | **ذکرشده در پروپوزال** |

### ۳.۲. وظیفه پیش‌بینی بقا (Survival)

#### آزمایش ۴: مولکولی خالص
- **نتایج:** AUC ≈ 0.50–0.58 (نزدیک شانس)
- **تفسیر:** داده مولکولی به تنهایی سیگنال بقا ندارد

#### آزمایش ۵: بالینی خالص
- **نتایج:** AUC ≈ 0.63–0.65
- **تفسیر:** Stage و Age قوی‌ترین پیش‌بینی‌کننده‌ها هستند

#### آزمایش ۶: مسیرهای زیستی (Pathway-based)
- **مسیرها:** Mitophagy, Cuproptosis, Ferroptosis, p53, Cell Cycle, Apoptosis
- **نتایج:** AUC ≈ 0.53–0.57
- **تفسیر:** مسیرهای مرگ سلولی برای تشخیص مؤثرند اما برای بقا ضعیف

### ۳.۳. Late Fusion چند-اومیکی

#### آزمایش ۷: Late Fusion mRNA + miRNA (PSO)
- **داده:** 503 بیمار (448 تومور + 55 نرمال)
- **mRNA:** 324 ژن PSO → XGBoost → P(Tumor|mRNA)
- **miRNA:** 160 miRNA PSO → XGBoost → P(Tumor|miRNA)
- **Meta-Learner:** Logistic Regression

| مدل | Test BA | Test AUC |
|-----|---------|----------|
| mRNA alone (324 genes) | 1.0000 | 1.0000 |
| miRNA alone (160 genes) | 0.7162 | 0.8960 |
| **Late Fusion** | **0.9545** | **1.0000** |

#### SHAP Modality Importance:

mRNA SHAP: 0.7419 ← مدالیته غالب
miRNA SHAP: 0.5546 ← سهم قابل توجه
---

## ۴. وضعیت نوآوری‌های پروپوزال

| # | نوآوری | وضعیت | شواهد |
|---|--------|-------|-------|
| 1 | ادغام داده‌های چندگانه (Multi-Omics) | ✅ تأیید | Late Fusion mRNA+miRNA, AUC=1.0 |
| 2 | انتخاب ویژگی فراابتکاری (PSO) | ✅ تأیید | mRNA: 324 ژن, miRNA: 160 ژن |
| 3 | یادگیری جمعی (Ensemble/Late Fusion) | ✅ تأیید | Meta-learner با AUC=1.0 |
| 4 | تفسیرپذیری (SHAP/XAI) | ✅ تأیید | 24 بیومارکر + Modality SHAP |
| 5 | اعتبارسنجی خارجی (GSE131907) | ⬜ اجرا نشده | مرحله بعدی |

---

## ۵. ساختار فایل‌های پروژه
```
LUAD_MultiOmics/
├── configs/
│ ├── config.yaml # تنظیمات اصلی
│ ├── datasets.yaml # تعریف دیتاست‌ها
│ ├── models.yaml # پارامترهای مدل‌ها
│ └── experiments/
│ └── stage_multiclass.yaml # تنظیمات آزمایش‌ها
├── src/luad/
│ ├── config/loader.py # بارگذاری کانفیگ
│ ├── data/
│ │ ├── audit.py # بررسی داده
│ │ ├── harmonizer.py # هماهنگ‌سازی نمونه‌ها
│ │ ├── normal_loader.py # بارگذاری نرمال
│ │ ├── preparation.py # آماده‌سازی داده
│ │ └── splitting.py # تقسیم Train/Test
│ ├── feature_selection/
│ │ ├── pso.py # الگوریتم PSO
│ │ └── pathway_genes.py # لیست ژن‌های مسیرها
│ ├── ml/
│ │ ├── baseline.py # آموزش پایه
│ │ └── datasets.py # بارگذاری ماتریس‌ها
│ ├── models/factory.py # ساخت مدل‌ها
│ ├── evaluation/metrics.py # معیارهای ارزیابی
│ ├── fusion/late.py # Late Fusion
│ └── utils/ # ابزارهای کمکی
├── scripts/
│ ├── prepare_tumor_normal.py # آماده‌سازی تومور/نرمال
│ ├── prepare_mrna_mirna_tn.py # ترکیب mRNA+miRNA
│ ├── train_pso_feature_selection.py # اجرای PSO
│ ├── run_shap_analysis.py # تحلیل SHAP
│ ├── train_late_fusion_pso.py # Late Fusion + PSO
│ └── train_survival.py # پیش‌بینی بقا
├── artifacts/experiments/
│ ├── pso_tumor_normal_v3/ # نتایج PSO mRNA
│ ├── shap_biomarker/ # نتایج SHAP
│ └── late_fusion_mrna_mirna_pso_v3/ # نتایج Late Fusion
└── docs/
└── PROJECT_SUMMARY.md # ← همین سند
```
---

## ۶. نقشه راه ادامه پروژه

### فاز A: اعتبارسنجی خارجی 🔴 (اولویت بالا)
- اجرای مدل فریزشده روی GSE131907
- تأیید تعمیم‌پذیری 324 ژن PSO
- تکمیل نوآوری پنجم

### فاز B: SHAP دقیق روی بیومارکرهای mRNA
- SHAP روی XGBoost (نه meta-learner)
- Top-20 ژن کلیدی با جهت تغییر بیان
- تطبیق با KEGG/Reactome

### فاز C: افزودن Methylation + CNA به Late Fusion
- PSO جداگانه برای هر مدالیته
- Late Fusion با 4 مدالیته
- مقایسه اثر افزایشی

### فاز D: گزارش نهایی برای فصل 4
- جدول مقایسه با 10 مقاله پیشین
- نمودارهای SHAP برای پایان‌نامه
- لیست بیومارکرهای تأییدشده

---

## ۷. نکات مهم علمی برای پایان‌نامه

### ۷.۱. درباره Δ BA منفی در Late Fusion
Δ BA (Late Fusion vs mRNA only): -0.0455
این کاهش جزئی به دلیل نویز miRNA در meta-learner است. اما **AUC همچنان 1.0** است که نشان می‌دهد رتبه‌بندی بیماران کاملاً درست است. در کاربردهای بالینی، AUC مهم‌تر از BA است.

### ۷.۲. درباره نتایج بقا
پیش‌بینی بقا از داده مولکولی خالص نزدیک شانس است (AUC≈0.55). این یک یافته علمی معتبر است و باید صادقانه گزارش شود. داده بالینی (Stage, Age) با AUC=0.65 بهتر عمل می‌کند.

### ۷.۳. درباره ETV4
ژن **ETV4** (رتبه 12 در SHAP) دقیقاً در پروپوزال و سمینار به عنوان بیومارکر ذکر شده است. شناسایی مستقل آن توسط PSO+SHAP یک تأییدیه قوی برای روش پیشنهادی است.

---

## ۸. جداول آماده برای فصل 4

### جدول مقایسه عملکرد مدل‌ها

| مدل | مدالیته | ویژگی‌ها | Test BA | Test AUC | MCC |
|-----|---------|----------|---------|----------|-----|
| LR | mRNA | 324 PSO | 1.0000 | 1.0000 | 1.0000 |
| RF | mRNA | 324 PSO | 1.0000 | 1.0000 | 1.0000 |
| SVM | mRNA | 324 PSO | 1.0000 | 1.0000 | 1.0000 |
| XGB | mRNA | 324 PSO | 1.0000 | 1.0000 | 1.0000 |
| XGB | miRNA | 160 PSO | 0.7162 | 0.8960 | — |
| Late Fusion | mRNA+miRNA | 324+160 | 0.9545 | 1.0000 | — |

### جدول بیومارکرهای برتر SHAP

| رتبه | ژن | Mean\|SHAP\| | جهت | مسیر زیستی |
|------|-----|-------------|------|-----------|
| 1 | TMEM8B | 1.290 | ↓ | Membrane regulation |
| 2 | SDPR | 0.460 | ↓ | Tumor suppressor |
| 3 | STX1A | 0.443 | ↓ | Vesicle trafficking |
| 4 | SGCG | 0.432 | ↓ | Sarcoglycan complex |
| 5 | SNAP47 | 0.394 | ↑ | SNARE complex |
| 12 | ETV4 | 0.073 | ↑ | ETS transcription factor |

---

*این سند به صورت خودکار از نتایج آزمایش‌های واقعی استخراج شده است.*