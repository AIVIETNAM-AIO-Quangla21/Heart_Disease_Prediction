"""
Heart Disease Prediction — EDA & Logistic Regression
=====================================================
Phân tích khám phá dữ liệu (EDA) và huấn luyện mô hình Logistic Regression
để dự đoán nguy cơ mắc bệnh tim, sử dụng bộ dữ liệu UCI Heart Disease.

Cách chạy:
    python src/heart_disease_analysis.py

(File này dùng cú pháp cell "# %%" nên cũng có thể chạy từng ô trong
VS Code / Jupyter / Spyder.)
"""

# %% 1. Import thư viện
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, r2_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# %% 2. Nạp dữ liệu
# Đường dẫn tương đối theo cấu trúc repo: <root>/data/heart_disease_uci.csv
DATA_PATH = r"C:\Users\ADMIN\Documents\heart_disease_uci.csv"  # đổi đường dẫn nếu file ở vị trí khác
df = pd.read_csv(DATA_PATH)

# %% 3. Tổng quan dữ liệu
print("Kích thước dữ liệu:", df.shape)
print("\nKiểu dữ liệu từng cột:")
print(df.dtypes)
print("\n5 dòng đầu tiên:")
print(df.head())

# %% 4. Kiểm tra giá trị thiếu
missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df) * 100).round(2)
missing_report = pd.concat(
    [missing_count, missing_pct], axis=1, keys=["so_luong_thieu", "ty_le_thieu_%"]
)
missing_report = missing_report[missing_report["so_luong_thieu"] > 0].sort_values(
    "ty_le_thieu_%", ascending=False
)
print("\nBáo cáo giá trị thiếu:")
print(missing_report)

if len(missing_report):
    plt.figure(figsize=(9, 4.5))
    sns.barplot(
        x=missing_report["ty_le_thieu_%"],
        y=missing_report.index,
        color="#B33951",
    )
    plt.xlabel("% giá trị thiếu")
    plt.ylabel("")
    plt.title("Tỷ lệ giá trị thiếu theo từng cột")
    plt.tight_layout()
    plt.show()

# Giá trị thiếu có thể liên quan đến từng trung tâm thu thập dữ liệu
if "dataset" in df.columns:
    completeness_by_site = (
        1 - df.groupby("dataset").apply(lambda g: g.isnull().mean().mean())
    ) * 100
    print("\nĐộ đầy đủ dữ liệu trung bình theo trung tâm (%):")
    print(completeness_by_site.round(1))

# %% 5. Thống kê mô tả
print("\nThống kê mô tả các biến số:")
print(df.describe())

print("\nThống kê mô tả các biến phân loại:")
print(df.select_dtypes(exclude="number").describe())

# %% 6. Xử lý & tạo biến mục tiêu nhị phân
# num: 0 = không bệnh, 1-4 = có bệnh với mức độ tăng dần
# -> tạo thêm cột 'target' nhị phân để dễ phân tích/mô hình hoá sau này
df["target"] = (df["num"] > 0).astype(int)

target_counts = df["target"].value_counts().sort_index()
print("\nPhân bố biến mục tiêu (0 = không bệnh, 1 = có bệnh):")
print(target_counts)
print("Tỷ lệ có bệnh: {:.1f}%".format(df["target"].mean() * 100))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.countplot(x="num", data=df, hue="num", palette="rocket", legend=False, ax=axes[0])
axes[0].set_title("Mức độ hẹp động mạch (num: 0-4)")
axes[0].set_xlabel("num")

sns.countplot(x="target", data=df, hue="target", palette=["#2A6F77", "#B33951"], legend=False, ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Không bệnh", "Có bệnh"])
axes[1].set_title("Phân bố biến mục tiêu nhị phân")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()
plt.close(fig)

# %% 7. Phân tích đơn biến — biến số (numerical)
num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
num_cols = [c for c in num_cols if c in df.columns]

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[col].dropna(), kde=True, color="#2A6F77", ax=axes[0])
    axes[0].axvline(df[col].mean(), color="#B33951", linestyle="--", label="Trung bình")
    axes[0].set_title(f"Phân bố của {col}")
    axes[0].legend()

    sns.boxplot(x=df[col].dropna(), color="#C98A2B", ax=axes[1])
    axes[1].set_title(f"Boxplot của {col} (phát hiện outlier)")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

# %% 8. Phân tích đơn biến — biến phân loại (categorical)
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "dataset"]
cat_cols = [c for c in cat_cols if c in df.columns]

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order, color="#2A6F77")
    plt.title(f"Phân bố của biến '{col}'")
    plt.xlabel("Số lượng")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    plt.close()

# %% 9. Phân tích hai biến — biến số vs biến mục tiêu
for col in num_cols:
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(
        x="target", y=col, data=df, hue="target",
        palette=["#2A6F77", "#B33951"], legend=False,
    )
    plt.xticks([0, 1], ["Không bệnh", "Có bệnh"])
    plt.title(f"{col} theo tình trạng bệnh")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()
    plt.close()

# %% 10. Phân tích hai biến — biến phân loại vs biến mục tiêu
for col in cat_cols:
    ct = pd.crosstab(df[col], df["target"], normalize="index") * 100
    ct = ct.sort_values(1, ascending=False) if 1 in ct.columns else ct

    ax = ct.plot(
        kind="barh", stacked=True, color=["#2A6F77", "#B33951"], figsize=(8, 4.5)
    )
    plt.title(f"Tỷ lệ mắc bệnh theo '{col}' (%)")
    plt.xlabel("% bệnh nhân")
    plt.ylabel("")
    plt.legend(["Không bệnh", "Có bệnh"], loc="lower right")
    plt.tight_layout()
    plt.show()
    plt.close()

# %% 11. Ma trận tương quan giữa các biến số
corr_cols = num_cols + ["num"]
corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Ma trận tương quan giữa các biến số")
plt.tight_layout()
plt.show()

print("\nTương quan của từng biến với mức độ bệnh (num), sắp xếp giảm dần:")
print(corr["num"].drop("num").sort_values(ascending=False))

# %% 12. Pairplot cho các biến số quan trọng theo tình trạng bệnh
important_cols = [c for c in ["age", "trestbps", "chol", "thalch", "oldpeak"] if c in df.columns]
sns.pairplot(
    df[important_cols + ["target"]].dropna(),
    hue="target",
    palette=["#2A6F77", "#B33951"],
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 25},
)
plt.suptitle("Pairplot các biến số quan trọng theo tình trạng bệnh", y=1.02)
plt.show()

# %% 13. Phát hiện outlier bằng phương pháp IQR
print("\nPhát hiện outlier (phương pháp IQR 1.5x):")
outlier_summary = {}
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_summary[col] = len(outliers)
    print(f"  {col}: {len(outliers)} outlier(s)  (ngưỡng hợp lệ: [{lower:.1f}, {upper:.1f}])")

# %% 14. Tổng kết nhanh
print("\n" + "=" * 60)
print("TÓM TẮT EDA")
print("=" * 60)
print(f"- Tổng số bệnh nhân     : {len(df)}")
print(f"- Số biến               : {df.shape[1]}")
print(f"- Tỷ lệ có bệnh         : {df['target'].mean() * 100:.1f}%")
print(f"- Tuổi trung bình       : {df['age'].mean():.1f} (min {df['age'].min()}, max {df['age'].max()})")
if len(missing_report):
    top_missing = missing_report.index[0]
    print(f"- Cột thiếu nhiều nhất  : {top_missing} ({missing_report.iloc[0, 1]:.0f}%)")
top_corr = corr["num"].drop("num").abs().sort_values(ascending=False).index[0]
print(f"- Biến tương quan mạnh nhất với 'num': {top_corr} (r = {corr['num'][top_corr]:.2f})")
print("=" * 60)

# ================================
# Phân tích và huấn luyện mô hình
# =================================

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
TEAL, CRIMSON, GOLD = "#2A6F77", "#B33951", "#C98A2B"

RANDOM_STATE = 42
# num: 0 = không bệnh, 1-4 = có bệnh (mức độ tăng dần)
# -> chuyển thành bài toán phân loại nhị phân: có bệnh hay không
df["target"] = (df["num"] > 0).astype(int)

# Loại 'id' (không mang thông tin), 'dataset' (chỉ là nơi thu thập dữ liệu,
# không phải yếu tố lâm sàng — giữ lại dễ khiến mô hình học "đặc điểm bệnh
# viện" thay vì đặc điểm bệnh lý thật) và 'num' (chính là nguồn tạo ra target,
# giữ lại sẽ gây rò rỉ dữ liệu - data leakage).
drop_cols = ["id", "dataset", "num", "target"]
X = df.drop(columns=drop_cols)
y = df["target"]

numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

print("Số đặc trưng đầu vào:", X.shape[1])
print("Phân bố target:\n", y.value_counts(normalize=True).round(3))

# %% 3. Chia tập huấn luyện / kiểm tra (giữ nguyên tỷ lệ lớp - stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTập huấn luyện: {X_train.shape[0]} mẫu | Tập kiểm tra: {X_test.shape[0]} mẫu")

# %% 4. Pipeline tiền xử lý
# - Biến số: điền khuyết bằng median, sau đó chuẩn hoá (StandardScaler)
# - Biến phân loại: điền khuyết bằng giá trị xuất hiện nhiều nhất, sau đó one-hot encode
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# %% 5. Xây dựng & huấn luyện mô hình Logistic Regression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

model.fit(X_train, y_train)
print("\nĐã huấn luyện xong mô hình Logistic Regression.")

# %% 6. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # xác suất thuộc lớp "có bệnh"

# %% 7. Đánh giá mô hình — các chỉ số tổng quát
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 55)
print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP KIỂM TRA")
print("=" * 55)
print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}  (trong số dự đoán 'có bệnh', bao nhiêu % đúng)")
print(f"Recall    : {rec:.3f}  (trong số bệnh nhân thật sự có bệnh, mô hình bắt được bao nhiêu %)")
print(f"F1-score  : {f1:.3f}")
print(f"AUC-ROC   : {auc:.3f}")
print("\nBáo cáo chi tiết theo từng lớp:")
print(classification_report(y_test, y_pred, target_names=["Không bệnh", "Có bệnh"]))

# %% 8. Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5.5, 4.8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="RdBu_r", cbar=False,
    xticklabels=["Không bệnh", "Có bệnh"],
    yticklabels=["Không bệnh", "Có bệnh"],
    annot_kws={"fontsize": 13, "fontweight": "bold"},
)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn (Confusion Matrix)")
plt.tight_layout()
plt.show()
plt.close()

tn, fp, fn, tp = cm.ravel()
print(f"\nDương tính thật (TP): {tp}  |  Âm tính thật (TN): {tn}")
print(f"Dương tính giả (FP) : {fp}  |  Âm tính giả (FN)  : {fn}")

# %% 9. Đường cong ROC & AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6.5, 5.5))
plt.plot(fpr, tpr, color=CRIMSON, linewidth=2.2, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.3, label="Mô hình ngẫu nhiên")
plt.xlabel("Tỷ lệ dương tính giả (False Positive Rate)")
plt.ylabel("Tỷ lệ dương tính thật (True Positive Rate)")
plt.title("Đường cong ROC")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
plt.close()

# %% 10. Đường cong Precision-Recall
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure(figsize=(6.5, 5.5))
plt.plot(rec_curve, prec_curve, color=TEAL, linewidth=2.2, label=f"Average Precision = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Đường cong Precision–Recall")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
plt.close()

# %% 11. Kiểm định chéo (Cross-Validation) — đánh giá độ ổn định của mô hình
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print("\n" + "=" * 55)
print("KIỂM ĐỊNH CHÉO 5-FOLD (trên toàn bộ dữ liệu)")
print("=" * 55)
print(f"Accuracy trung bình : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print(f"AUC-ROC trung bình  : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

# %% 12. Diễn giải mô hình — hệ số hồi quy & odds ratio
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
coefs = model.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "dac_trung": feature_names,
    "he_so": coefs,
    "odds_ratio": np.exp(coefs),
}).sort_values("he_so", key=np.abs, ascending=False)

print("\nHệ số hồi quy (sắp xếp theo độ ảnh hưởng, |hệ số| giảm dần):")
print(coef_df.to_string(index=False))

plt.figure(figsize=(9, 6.5))
plot_df = coef_df.sort_values("he_so")
colors = [CRIMSON if c > 0 else TEAL for c in plot_df["he_so"]]
plt.barh(plot_df["dac_trung"], plot_df["he_so"], color=colors)
plt.axvline(0, color="#374151", linewidth=0.8)
plt.xlabel("Hệ số hồi quy (ảnh hưởng đến log-odds mắc bệnh)")
plt.title("Mức độ ảnh hưởng của từng đặc trưng\n(đỏ = tăng nguy cơ, xanh = giảm nguy cơ)")
plt.tight_layout()
plt.show()
plt.close()

# %% 13. Ví dụ minh hoạ: dự đoán cho một bệnh nhân mới
benh_nhan_moi = pd.DataFrame([{
    "age": 58, "sex": "Male", "cp": "asymptomatic", "trestbps": 145,
    "chol": 270, "fbs": True, "restecg": "lv hypertrophy", "thalch": 120,
    "exang": True, "oldpeak": 2.0, "slope": "flat", "ca": 2.0, "thal": "reversable defect",
}])

pred = model.predict(benh_nhan_moi)[0]
proba = model.predict_proba(benh_nhan_moi)[0, 1]
print("\n" + "=" * 55)
print("VÍ DỤ: DỰ ĐOÁN CHO MỘT BỆNH NHÂN MỚI")
print("=" * 55)
print(f"Kết quả dự đoán  : {'CÓ nguy cơ bệnh tim' if pred == 1 else 'KHÔNG có nguy cơ bệnh tim'}")
print(f"Xác suất mắc bệnh: {proba * 100:.1f}%")

# %% 14. Tóm tắt kết quả
print("\n" + "=" * 55)
print("TÓM TẮT")
print("=" * 55)
print(f"- Mô hình         : Logistic Regression")
print(f"- Số đặc trưng     : {X.shape[1]} (sau one-hot: {len(feature_names)})")
print(f"- Accuracy (test)  : {acc:.3f}")
print(f"- AUC-ROC (test)   : {auc:.3f}")
print(f"- Accuracy (CV 5-fold): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
top_feature = coef_df.iloc[0]
huong = "tăng" if top_feature["he_so"] > 0 else "giảm"
print(f"- Đặc trưng ảnh hưởng mạnh nhất: {top_feature['dac_trung']} ({huong} nguy cơ mắc bệnh)")
print("=" * 55)
