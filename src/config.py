from pathlib import Path

# ===== Paths =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Assignment1_mimic dataset.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
CM_DIR = RESULTS_DIR / "confusion_matrices"
ROC_DIR = RESULTS_DIR / "roc_curves"

# ===== Target =====
# 建议本作业预测 ICU death
TARGET_COLUMN = "icu_death_flag"

# ===== Columns to drop =====
# 明显的 ID 列
DROP_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
]

# 明显有信息泄露风险 or 事后才知道的列
LEAKAGE_COLUMNS = [
    "hospital_expire_flag",  # 与 ICU death 高度相关，属于结局信息
    "deathtime",             # 明确结局后才知道
    "outtime",               # 出 ICU 时间，事后信息
    "los",                   # 住院/ICU 时长，事后信息
    "last_careunit",         # 最终 care unit，事后信息
    "intime",                # 暂时不解析时间，先删除避免无意义字符串
]

# ===== Split =====
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

# ===== Feature selection =====
MISSING_THRESHOLD = 0.40      # 缺失率 > 40% 删除
LOW_VARIANCE_THRESHOLD = 0.0  # 近似常量列删除
CORR_THRESHOLD = 0.90         # 高相关阈值
TOP_K_FEATURES = 30           # 第三阶段最多保留 30 个特征

# ===== CV / reporting =====
CV_FOLDS = 5