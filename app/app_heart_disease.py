"""
Heart Disease Prediction — Streamlit App
=========================================
Ứng dụng tương tác cho phép:
  1. Nhập chỉ số lâm sàng của một bệnh nhân và dự đoán nguy cơ mắc bệnh tim.
  2. Khám phá dữ liệu (EDA) trực quan.
  3. Xem hiệu năng & cách diễn giải mô hình Logistic Regression.

Chạy:
    streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

# --------------------------------------------------------------------------
# Cấu hình chung
# --------------------------------------------------------------------------

RANDOM_STATE = 42

TEAL, CRIMSON, GOLD = "#2A6F77", "#B33951", "#C98A2B"
INK, PAPER = "#1F2937", "#F7F5F0"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

CP_LABELS = {
    "typical angina": "Đau thắt ngực điển hình",
    "atypical angina": "Đau thắt ngực không điển hình",
    "non-anginal": "Đau ngực không do tim",
    "asymptomatic": "Không triệu chứng",
}
RESTECG_LABELS = {
    "normal": "Bình thường",
    "lv hypertrophy": "Phì đại thất trái",
    "st-t abnormality": "Bất thường ST-T",
}
SLOPE_LABELS = {
    "upsloping": "Dốc lên",
    "flat": "Phẳng",
    "downsloping": "Dốc xuống",
}
THAL_LABELS = {
    "normal": "Bình thường",
    "fixed defect": "Khiếm khuyết cố định",
    "reversable defect": "Khiếm khuyết có thể hồi phục",
}

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Style — CSS tối giản, theo bảng màu của dự án (teal / crimson / gold)
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAPER}; }}
    h1, h2, h3 {{ color: {INK}; font-family: 'Georgia', serif; }}
    .risk-card {{
        padding: 1.4rem 1.6rem; border-radius: 14px; color: white;
        text-align: center; margin-bottom: 0.8rem;
    }}
    .risk-card h2 {{ color: white; margin: 0; font-size: 2.2rem; }}
    .risk-card p {{ margin: 0.2rem 0 0 0; opacity: 0.92; }}
    .metric-box {{
        background: white; border-radius: 10px; padding: 0.9rem 1rem;
        border: 1px solid #E5E1D8;
    }}
    .footer-note {{
        font-size: 0.82rem; color: #6B7280; border-top: 1px solid #E5E1D8;
        padding-top: 0.8rem; margin-top: 1.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Dữ liệu & Mô hình (cache để không phải tính lại mỗi lần tương tác)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ADMIN\PycharmProjects\PythonProject1\data\heart_disease_uci.csv")
    df["target"] = (df["num"] > 0).astype(int)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    drop_cols = ["id", "dataset", "num", "target"]
    X = df.drop(columns=drop_cols)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    coefs = model.named_steps["classifier"].coef_[0]
    coef_df = pd.DataFrame({
        "dac_trung": feature_names, "he_so": coefs, "odds_ratio": np.exp(coefs),
    }).sort_values("he_so")

    return {
        "model": model, "metrics": metrics, "cm": cm, "fpr": fpr, "tpr": tpr,
        "cv_acc": cv_acc, "cv_auc": cv_auc, "coef_df": coef_df,
        "X": X, "y": y,
    }


df = load_data()
artifacts = train_model(df)
model = artifacts["model"]

# --------------------------------------------------------------------------
# Sidebar — nhập thông tin bệnh nhân
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🩺 Thông tin bệnh nhân")
st.sidebar.caption("Điều chỉnh các chỉ số rồi bấm **Dự đoán** ở cuối form.")

with st.sidebar.expander("👤 Thông tin chung", expanded=True):
    age = st.slider("Tuổi", 18, 100, 54)
    sex = st.radio("Giới tính", ["Male", "Female"], horizontal=True,
                    format_func=lambda v: "Nam" if v == "Male" else "Nữ")

with st.sidebar.expander("💓 Triệu chứng & đo lường", expanded=True):
    cp = st.selectbox("Loại đau ngực", list(CP_LABELS.keys()),
                       format_func=lambda v: CP_LABELS[v], index=3)
    trestbps = st.slider("Huyết áp lúc nghỉ (mm Hg)", 80, 220, 130)
    chol = st.slider("Cholesterol huyết thanh (mg/dl)", 100, 600, 246)
    fbs = st.checkbox("Đường huyết đói > 120 mg/dl", value=False)
    thalch = st.slider("Nhịp tim tối đa đạt được", 60, 220, 150)
    exang = st.checkbox("Đau thắt ngực khi gắng sức", value=False)

with st.sidebar.expander("🧪 Cận lâm sàng", expanded=True):
    restecg = st.selectbox("Điện tâm đồ lúc nghỉ", list(RESTECG_LABELS.keys()),
                            format_func=lambda v: RESTECG_LABELS[v], index=0)
    oldpeak = st.slider("Độ chênh ST do gắng sức (oldpeak)", -2.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Độ dốc đoạn ST", list(SLOPE_LABELS.keys()),
                          format_func=lambda v: SLOPE_LABELS[v], index=1)
    ca = st.selectbox("Số mạch máu chính bị hẹp", [0, 1, 2, 3], index=0)
    thal = st.selectbox("Kết quả thalassemia", list(THAL_LABELS.keys()),
                         format_func=lambda v: THAL_LABELS[v], index=0)

predict_clicked = st.sidebar.button("🔮 Dự đoán", width="stretch", type="primary")

patient = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": float(ca), "thal": thal,
}])

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🫀 Heart Disease Prediction")
st.caption(
    "Dự đoán nguy cơ mắc bệnh tim từ chỉ số lâm sàng, huấn luyện trên bộ dữ liệu "
    "UCI Heart Disease (920 bệnh nhân — Cleveland, Hungary, Switzerland, VA Long Beach)."
)

tab_predict, tab_eda, tab_model = st.tabs([
    "🔮 Dự đoán", "📊 Khám phá dữ liệu", "📈 Hiệu năng mô hình",
])

# --------------------------------------------------------------------------
# TAB 1 — Dự đoán
# --------------------------------------------------------------------------
with tab_predict:
    proba = model.predict_proba(patient)[0, 1]

    if proba < 0.30:
        color, level = "#2A9D6F", "Nguy cơ THẤP"
    elif proba < 0.60:
        color, level = GOLD, "Nguy cơ TRUNG BÌNH"
    else:
        color, level = CRIMSON, "Nguy cơ CAO"

    col_gauge, col_info = st.columns([1, 1.4])

    with col_gauge:
        st.markdown(
            f"""<div class="risk-card" style="background:{color};">
                    <p>Xác suất mắc bệnh tim</p>
                    <h2>{proba * 100:.1f}%</h2>
                    <p><b>{level}</b></p>
                </div>""",
            unsafe_allow_html=True,
        )

        # Biểu đồ đồng hồ đo (gauge) nửa hình tròn
        fig, ax = plt.subplots(figsize=(4.2, 2.4), subplot_kw={"aspect": "equal"})
        theta = np.linspace(np.pi, 0, 100)
        bands = [(0, 0.30, "#2A9D6F"), (0.30, 0.60, GOLD), (0.60, 1.0, CRIMSON)]
        for lo, hi, c in bands:
            t = np.linspace(np.pi * (1 - lo), np.pi * (1 - hi), 30)
            ax.plot(np.cos(t), np.sin(t), color=c, linewidth=14, solid_capstyle="butt")
        needle_angle = np.pi * (1 - proba)
        ax.plot([0, 0.85 * np.cos(needle_angle)], [0, 0.85 * np.sin(needle_angle)],
                 color=INK, linewidth=2.5)
        ax.scatter([0], [0], color=INK, s=40, zorder=5)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.axis("off")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_info:
        st.markdown("#### Diễn giải")
        if not predict_clicked:
            st.info("Điều chỉnh thông tin bên trái rồi bấm **Dự đoán** để cập nhật kết quả (kết quả bên dưới luôn phản ánh giá trị hiện tại của form).")
        st.write(
            f"Với các chỉ số đã nhập, mô hình ước tính bệnh nhân này có "
            f"**{proba * 100:.1f}%** khả năng mắc bệnh tim, xếp vào nhóm **{level.lower()}**."
        )

        avg_row = df[df["target"] == 1][NUMERIC_COLS].mean()
        compare_df = pd.DataFrame({
            "Chỉ số": ["Tuổi", "Huyết áp", "Cholesterol", "Nhịp tim tối đa", "Oldpeak", "Số mạch hẹp"],
            "Bệnh nhân này": [age, trestbps, chol, thalch, oldpeak, ca],
            "TB nhóm có bệnh": avg_row.values.round(1),
        })
        st.dataframe(compare_df, hide_index=True, width="stretch")

        st.caption(
            "⚠️ Đây là mô hình học thuật mang tính minh hoạ, **không thay thế chẩn đoán y khoa**. "
            "Vui lòng tham khảo ý kiến bác sĩ để được tư vấn chính xác."
        )

# --------------------------------------------------------------------------
# TAB 2 — Khám phá dữ liệu (EDA)
# --------------------------------------------------------------------------
with tab_eda:
    st.markdown("#### Tổng quan bộ dữ liệu")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số bệnh nhân", f"{len(df)}")
    c2.metric("Tỷ lệ có bệnh", f"{df['target'].mean() * 100:.1f}%")
    c3.metric("Tuổi trung bình", f"{df['age'].mean():.1f}")
    c4.metric("Số trung tâm dữ liệu", f"{df['dataset'].nunique()}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="target", data=df, hue="target", palette=[TEAL, CRIMSON],
                      legend=False, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Không bệnh", "Có bệnh"])
        ax.set_title("Phân bố biến mục tiêu")
        ax.set_xlabel("")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_b:
        corr = df[NUMERIC_COLS + ["num"]].corr()
        fig, ax = plt.subplots(figsize=(5.2, 4.3))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, ax=ax)
        ax.set_title("Ma trận tương quan")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("#### Biến số theo tình trạng bệnh")
    sel_num = st.selectbox("Chọn biến số:", NUMERIC_COLS, index=0, key="eda_num")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    sns.boxplot(x="target", y=sel_num, data=df, hue="target",
                palette=[TEAL, CRIMSON], legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Không bệnh", "Có bệnh"])
    ax.set_xlabel("")
    ax.set_title(f"{sel_num} theo tình trạng bệnh")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("#### Tỷ lệ mắc bệnh theo biến phân loại")
    sel_cat = st.selectbox("Chọn biến phân loại:", CATEGORICAL_COLS, index=1, key="eda_cat")
    ct = pd.crosstab(df[sel_cat], df["target"], normalize="index") * 100
    ct = ct.sort_values(1, ascending=False) if 1 in ct.columns else ct
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ct.plot(kind="barh", stacked=True, color=[TEAL, CRIMSON], ax=ax)
    ax.set_xlabel("% bệnh nhân")
    ax.set_ylabel("")
    ax.set_title(f"Tỷ lệ mắc bệnh theo '{sel_cat}'")
    ax.legend(["Không bệnh", "Có bệnh"], loc="lower right")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# --------------------------------------------------------------------------
# TAB 3 — Hiệu năng mô hình
# --------------------------------------------------------------------------
with tab_model:
    m = artifacts["metrics"]
    st.markdown("#### Chỉ số trên tập kiểm tra (20%)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1-score", f"{m['f1']:.3f}")
    c5.metric("AUC-ROC", f"{m['auc']:.3f}")

    st.markdown("#### Kiểm định chéo 5-fold (toàn bộ dữ liệu)")
    cv_acc, cv_auc = artifacts["cv_acc"], artifacts["cv_auc"]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy (CV)", f"{cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    c2.metric("AUC-ROC (CV)", f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        fig, ax = plt.subplots(figsize=(4.6, 4))
        sns.heatmap(artifacts["cm"], annot=True, fmt="d", cmap="RdBu_r", cbar=False,
                    xticklabels=["Không bệnh", "Có bệnh"],
                    yticklabels=["Không bệnh", "Có bệnh"],
                    annot_kws={"fontsize": 13, "fontweight": "bold"}, ax=ax)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title("Ma trận nhầm lẫn")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_roc:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(artifacts["fpr"], artifacts["tpr"], color=CRIMSON, linewidth=2.2,
                label=f"Logistic Regression (AUC={m['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.2,
                label="Mô hình ngẫu nhiên")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Đường cong ROC")
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("#### Mức độ ảnh hưởng của từng đặc trưng")
    coef_df = artifacts["coef_df"]
    colors = [CRIMSON if c > 0 else TEAL for c in coef_df["he_so"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coef_df["dac_trung"], coef_df["he_so"], color=colors)
    ax.axvline(0, color="#374151", linewidth=0.8)
    ax.set_xlabel("Hệ số hồi quy (log-odds)")
    ax.set_title("Đỏ = tăng nguy cơ · Xanh = giảm nguy cơ")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    with st.expander("Xem bảng hệ số & odds ratio đầy đủ"):
        st.dataframe(
            coef_df.rename(columns={
                "dac_trung": "Đặc trưng", "he_so": "Hệ số", "odds_ratio": "Odds ratio",
            }).sort_values("Hệ số", key=np.abs, ascending=False),
            hide_index=True, width="stretch",
        )

st.markdown(
    """<div class="footer-note">
    Dữ liệu: UCI Heart Disease Data Set (Cleveland, Hungary, Switzerland, VA Long Beach) ·
    Mô hình: Logistic Regression (scikit-learn) · Chỉ phục vụ mục đích học tập/minh hoạ.
    </div>""",
    unsafe_allow_html=True,
)
