from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 1. CẤU HÌNH ỨNG DỤNG & GIAO DIỆN (CSS)
# =========================================================

st.set_page_config(
    page_title="LOGI AI - Dự đoán giao hàng trễ",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }

        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 65%, #0f766e 100%);
            color: #ffffff;
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            margin-bottom: 1.25rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        }

        .hero-banner h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
        }

        .hero-banner .hero-sub {
            margin: 0.25rem 0 0.4rem 0;
            font-size: 1.05rem;
            font-weight: 500;
            color: #99f6e4;
        }

        .hero-banner .hero-desc {
            margin: 0;
            font-size: 0.92rem;
            color: #e2e8f0;
            opacity: 0.92;
        }

        .section-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #0d9488;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        }

        div[data-testid="stMetric"] label {
            color: #475569 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .risk-low {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }

        .risk-medium {
            background-color: #fef9c3;
            color: #854d0e;
            border: 1px solid #fde047;
        }

        .risk-high {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }

        .footer-box {
            text-align: center;
            color: #64748b;
            font-size: 0.88rem;
            padding: 0.6rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-banner">
        <h1>🚚 LOGI AI</h1>
        <div class="hero-sub">Hệ thống dự đoán khả năng giao hàng trễ</div>
        <p class="hero-desc">
            Ứng dụng Machine Learning sử dụng thuật toán Decision Tree
            để phân tích và dự đoán nguy cơ giao hàng trễ.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR /
    "Dữ liệu mẫu Food Delivery Orders & ETA Logistics.xlsx"
)

TARGET = "late_delivery"


# =========================================================
# 2. KHAI BÁO CÁC ĐẶC TRƯNG
# =========================================================

categorical_features = [
    "city",
    "delivery_area",
    "customer_type",
    "restaurant_type",
    "restaurant_primary_category",
    "weather",
    "traffic_level"
]

numeric_features = [
    "order_hour",
    "is_weekend",
    "restaurant_rating",
    "items_count",
    "distance_km",
    "delivery_partner_experience_months",
    "delivery_partner_rating",
    "restaurant_preparation_time_minutes",
    "estimated_delivery_time_minutes"
]

FEATURES = categorical_features + numeric_features

REQUIRED_COLUMNS = FEATURES + [TARGET]


# =========================================================
# 3. ĐỌC DỮ LIỆU
# =========================================================

@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path, engine="openpyxl")
    return data


def prepare_data(data):
    data = data.copy()

    # Kiểm tra các cột bắt buộc
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "File Excel đang thiếu các cột: "
            + ", ".join(missing_columns)
        )

    # Chuyển các cột số về dạng numeric
    for col in numeric_features + [TARGET]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # Thay giá trị vô cực bằng NaN
    data[numeric_features] = data[numeric_features].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Loại bỏ dòng không có nhãn mục tiêu
    data = data.dropna(subset=[TARGET])

    # Chỉ giữ nhãn 0 và 1
    data = data[data[TARGET].isin([0, 1])]

    data[TARGET] = data[TARGET].astype(int)

    if data.empty:
        raise ValueError(
            "Không có dữ liệu hợp lệ trong cột late_delivery."
        )

    if data[TARGET].nunique() < 2:
        raise ValueError(
            "Cột late_delivery phải có cả hai giá trị 0 và 1."
        )

    return data


# Cho phép đọc file cùng thư mục hoặc tải file lên
st.sidebar.markdown("### 📂 Nguồn dữ liệu")
uploaded_file = st.sidebar.file_uploader(
    "Tải lên file dữ liệu Excel nếu cần",
    type=["xlsx"]
)

try:
    if uploaded_file is not None:
        raw_df = pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )
        st.sidebar.success("Đã nhận file Excel tải lên.")

    elif DATA_PATH.exists():
        raw_df = load_data(str(DATA_PATH))
        st.sidebar.success("Đã đọc dữ liệu từ thư mục dự án.")

    else:
        st.error("Không tìm thấy file dữ liệu Excel.")

        st.write("Ứng dụng đang tìm file tại:")
        st.code(str(DATA_PATH))

        st.info(
            "Hãy đặt file Excel cùng thư mục với app.py "
            "hoặc tải file lên bằng mục bên trái."
        )

        st.stop()

    df = prepare_data(raw_df)

except Exception as error:
    st.error("Không thể đọc hoặc xử lý dữ liệu.")
    st.exception(error)
    st.stop()


# =========================================================
# 4. XÂY DỰNG MÔ HÌNH DECISION TREE
# =========================================================

X = df[FEATURES]
y = df[TARGET]

class_counts = y.value_counts()

if class_counts.min() < 2:
    st.error(
        "Mỗi lớp dữ liệu cần có ít nhất 2 mẫu "
        "để thực hiện chia tập huấn luyện và kiểm tra."
    )
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Xử lý dữ liệu phân loại
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# Xử lý dữ liệu số
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Kết hợp tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
        (
            "numeric",
            numeric_transformer,
            numeric_features
        )
    ]
)


# Xây dựng Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# Huấn luyện mô hình
model.fit(X_train, y_train)


# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)


# =========================================================
# 5. ĐÁNH GIÁ MÔ HÌNH
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)


# =========================================================
# 6. MENU ĐIỀU HƯỚNG
# =========================================================

st.sidebar.divider()

page = st.sidebar.radio(
    "CHỨC NĂNG",
    [
        "Tổng quan",
        "Dự đoán giao hàng",
        "Phân tích dữ liệu",
        "Thông tin mô hình"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    f"📦 Bản ghi hợp lệ: **{len(df):,}** | Đặc trưng: **{len(FEATURES)}**"
)


# =========================================================
# 7. TRANG TỔNG QUAN
# =========================================================

if page == "Tổng quan":

    st.header("📊 Tổng quan dữ liệu")

    total_orders = len(df)
    late_orders = int((df[TARGET] == 1).sum())
    ontime_orders = int((df[TARGET] == 0).sum())

    late_rate = late_orders / total_orders * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tổng số đơn hàng",
        f"{total_orders:,}"
    )

    col2.metric(
        "Đơn hàng giao trễ",
        f"{late_orders:,}"
    )

    col3.metric(
        "Đơn hàng không trễ",
        f"{ontime_orders:,}"
    )

    col4.metric(
        "Tỷ lệ giao trễ",
        f"{late_rate:.1f}%"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Tỷ lệ giao hàng")

        delivery_data = pd.DataFrame({
            "Trạng thái": [
                "Không trễ",
                "Giao trễ"
            ],
            "Số lượng": [
                ontime_orders,
                late_orders
            ]
        })

        fig = px.pie(
            delivery_data,
            names="Trạng thái",
            values="Số lượng",
            hole=0.45,
            title="Phân bố trạng thái giao hàng",
            color="Trạng thái",
            color_discrete_map={
                "Không trễ": "#0d9488",
                "Giao trễ": "#ef4444"
            }
        )
        fig.update_layout(
            margin=dict(t=45, b=20, l=10, r=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.subheader("Số lượng đơn hàng theo thời tiết")

        weather_data = (
            df.groupby(
                ["weather", TARGET]
            )
            .size()
            .reset_index(name="Số đơn")
        )

        weather_data["Trạng thái"] = weather_data[TARGET].map({
            0: "Không trễ",
            1: "Giao trễ"
        })

        fig = px.bar(
            weather_data,
            x="weather",
            y="Số đơn",
            color="Trạng thái",
            barmode="group",
            title="Thời tiết và trạng thái giao hàng",
            color_discrete_map={
                "Không trễ": "#0d9488",
                "Giao trễ": "#ef4444"
            }
        )
        fig.update_layout(
            margin=dict(t=45, b=20, l=10, r=10),
            xaxis_title="Thời tiết",
            yaxis_title="Số đơn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Dữ liệu mẫu")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# 8. TRANG DỰ ĐOÁN GIAO HÀNG
# =========================================================

elif page == "Dự đoán giao hàng":

    st.header("🔮 Dự đoán khả năng giao hàng trễ")

    st.info(
        "Nhập thông tin đơn hàng tại thời điểm điều phối giao hàng. "
        "Mô hình sẽ dự đoán khả năng đơn hàng bị giao trễ."
    )

    st.subheader("1. Thông tin đơn hàng")

    col1, col2 = st.columns(2)

    with col1:

        city = st.selectbox(
            "Thành phố",
            sorted(df["city"].dropna().unique())
        )

        delivery_area_options = sorted(
            df.loc[
                df["city"] == city,
                "delivery_area"
            ].dropna().unique()
        )

        delivery_area = st.selectbox(
            "Khu vực giao hàng",
            delivery_area_options
        )

        customer_type = st.selectbox(
            "Loại khách hàng",
            sorted(df["customer_type"].dropna().unique())
        )

        restaurant_type = st.selectbox(
            "Loại nhà hàng",
            sorted(df["restaurant_type"].dropna().unique())
        )

        restaurant_category = st.selectbox(
            "Danh mục nhà hàng",
            sorted(
                df["restaurant_primary_category"]
                .dropna()
                .unique()
            )
        )

    with col2:

        order_hour = st.number_input(
            "Giờ đặt hàng",
            min_value=0,
            max_value=23,
            value=12
        )

        is_weekend = st.selectbox(
            "Ngày cuối tuần",
            options=[0, 1],
            format_func=lambda x: "Có" if x == 1 else "Không"
        )

        weather = st.selectbox(
            "Thời tiết",
            sorted(df["weather"].dropna().unique())
        )

        traffic_level = st.selectbox(
            "Mức độ giao thông",
            sorted(df["traffic_level"].dropna().unique())
        )

        items_count = st.number_input(
            "Số lượng món",
            min_value=1,
            max_value=100,
            value=2
        )

    st.divider()

    st.subheader("2. Thông tin nhà hàng và tài xế")

    col3, col4 = st.columns(2)

    with col3:

        restaurant_rating = st.number_input(
            "Đánh giá nhà hàng",
            min_value=0.0,
            max_value=5.0,
            value=4.5,
            step=0.1
        )

        preparation_time = st.number_input(
            "Thời gian chuẩn bị món (phút)",
            min_value=1,
            max_value=300,
            value=20
        )

        distance_km = st.number_input(
            "Khoảng cách giao hàng (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

    with col4:

        driver_experience = st.number_input(
            "Kinh nghiệm tài xế (tháng)",
            min_value=0,
            max_value=500,
            value=12
        )

        driver_rating = st.number_input(
            "Đánh giá tài xế",
            min_value=0.0,
            max_value=5.0,
            value=4.5,
            step=0.1
        )

        estimated_time = st.number_input(
            "Thời gian giao hàng dự kiến (phút)",
            min_value=1,
            max_value=500,
            value=40
        )

    st.divider()

    if st.button(
        "🚀 DỰ ĐOÁN NGAY",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "city": city,
            "delivery_area": delivery_area,
            "customer_type": customer_type,
            "restaurant_type": restaurant_type,
            "restaurant_primary_category": restaurant_category,
            "weather": weather,
            "traffic_level": traffic_level,
            "order_hour": order_hour,
            "is_weekend": is_weekend,
            "restaurant_rating": restaurant_rating,
            "items_count": items_count,
            "distance_km": distance_km,
            "delivery_partner_experience_months": driver_experience,
            "delivery_partner_rating": driver_rating,
            "restaurant_preparation_time_minutes": preparation_time,
            "estimated_delivery_time_minutes": estimated_time
        }])

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        late_probability = probabilities[
            list(model.named_steps["classifier"].classes_).index(1)
        ]

        risk_pct = late_probability * 100
        if risk_pct < 30:
            badge_class = "risk-low"
            badge_text = "MỨC RỦI RO: THẤP (< 30%)"
        elif risk_pct < 70:
            badge_class = "risk-medium"
            badge_text = "MỨC RỦI RO: TRUNG BÌNH (30% - 70%)"
        else:
            badge_class = "risk-high"
            badge_text = "MỨC RỦI RO: CAO (> 70%)"

        st.divider()

        st.subheader("3. Kết quả dự đoán")

        col5, col6 = st.columns(2)

        with col5:

            if prediction == 1:
                st.error(
                    "⚠️ DỰ ĐOÁN: ĐƠN HÀNG CÓ NGUY CƠ GIAO TRỄ"
                )
            else:
                st.success(
                    "✅ DỰ ĐOÁN: ĐƠN HÀNG CÓ KHẢ NĂNG KHÔNG BỊ TRỄ"
                )

            st.markdown(
                f'<span class="risk-badge {badge_class}">{badge_text}</span>',
                unsafe_allow_html=True
            )

        with col6:

            st.metric(
                "Xác suất dự đoán giao trễ",
                f"{late_probability * 100:.2f}%"
            )
            st.progress(min(max(float(late_probability), 0.0), 1.0))

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=late_probability * 100,
                title={
                    "text": "Nguy cơ giao hàng trễ (%)"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "#0f172a"
                    },
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#D8F3DC"
                        },
                        {
                            "range": [30, 70],
                            "color": "#FFF3BF"
                        },
                        {
                            "range": [70, 100],
                            "color": "#FFD6D6"
                        }
                    ]
                }
            )
        )
        fig.update_layout(
            margin=dict(t=50, b=20, l=20, r=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if prediction == 1:
            st.warning(
                "Đề xuất: Theo dõi đơn hàng, kiểm tra tình trạng "
                "giao thông và phối hợp với nhà hàng hoặc tài xế "
                "để hạn chế nguy cơ giao trễ."
            )
        else:
            st.info(
                "Đơn hàng chưa được mô hình xác định là có nguy cơ "
                "giao trễ. Vẫn cần theo dõi quá trình giao hàng thực tế."
            )

        st.caption(
            "Lưu ý: Xác suất trên là kết quả từ mô hình Decision Tree, "
            "không phải cam kết về thời gian giao hàng thực tế."
        )


# =========================================================
# 9. TRANG PHÂN TÍCH DỮ LIỆU
# =========================================================

elif page == "Phân tích dữ liệu":

    st.header("📈 Phân tích dữ liệu")

    st.subheader("1. Phân tích theo mức độ giao thông")

    traffic_data = (
        df.groupby(
            ["traffic_level", TARGET]
        )
        .size()
        .reset_index(name="Số đơn")
    )

    traffic_data["Trạng thái"] = traffic_data[TARGET].map({
        0: "Không trễ",
        1: "Giao trễ"
    })

    fig = px.bar(
        traffic_data,
        x="traffic_level",
        y="Số đơn",
        color="Trạng thái",
        barmode="group",
        title="Ảnh hưởng của giao thông đến trạng thái giao hàng",
        color_discrete_map={
            "Không trễ": "#0d9488",
            "Giao trễ": "#ef4444"
        }
    )
    fig.update_layout(
        margin=dict(t=45, b=20, l=10, r=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("2. Phân tích theo khoảng cách giao hàng")

    fig = px.box(
        df,
        x=TARGET,
        y="distance_km",
        color=TARGET,
        labels={
            TARGET: "Trạng thái (0: Không trễ, 1: Trễ)",
            "distance_km": "Khoảng cách (km)"
        },
        title="Khoảng cách giao hàng và trạng thái giao hàng"
    )
    fig.update_layout(
        margin=dict(t=45, b=20, l=10, r=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("3. Phân tích thời gian chuẩn bị món")

    fig = px.box(
        df,
        x=TARGET,
        y="restaurant_preparation_time_minutes",
        color=TARGET,
        labels={
            TARGET: "Trạng thái (0: Không trễ, 1: Trễ)",
            "restaurant_preparation_time_minutes":
                "Thời gian chuẩn bị (phút)"
        },
        title="Thời gian chuẩn bị món và trạng thái giao hàng"
    )
    fig.update_layout(
        margin=dict(t=45, b=20, l=10, r=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("4. Bảng dữ liệu")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 Tải dữ liệu CSV",
        data=csv_data,
        file_name="food_delivery_data.csv",
        mime="text/csv"
    )


# =========================================================
# 10. TRANG THÔNG TIN MÔ HÌNH
# =========================================================

elif page == "Thông tin mô hình":

    st.header("🤖 Thông tin mô hình Machine Learning")

    st.subheader("1. Thuật toán sử dụng")

    st.write(
        """
        **Decision Tree Classifier – Cây quyết định**

        Thuật toán học máy có giám sát được sử dụng để phân loại
        đơn hàng thành hai nhóm:

- 0: Không giao trễ.
- 1: Giao trễ.

        Mô hình xây dựng các quy tắc phân nhánh dựa trên đặc trưng
        của đơn hàng để đưa ra dự đoán.
        """
    )

    st.subheader("2. Thông số mô hình")

    model_info = pd.DataFrame({
        "Thông số": [
            "Thuật toán",
            "Tiêu chí phân chia",
            "Độ sâu tối đa",
            "Số mẫu tối thiểu tại lá",
            "Xử lý mất cân bằng lớp",
            "Tỷ lệ huấn luyện",
            "Tỷ lệ kiểm tra"
        ],
        "Giá trị": [
            "Decision Tree Classifier",
            "Gini",
            "4",
            "2",
            "Balanced",
            "80%",
            "20%"
        ]
    })

    st.table(model_info)

    st.subheader("3. Kết quả đánh giá mô hình")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1-score",
        f"{f1 * 100:.2f}%"
    )

    st.caption(
        "Các chỉ số được tính trên tập kiểm tra 20%, "
        "với nhãn giao trễ (1) là lớp dương."
    )

    st.subheader("4. Ma trận nhầm lẫn")

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Thực tế: Không trễ",
            "Thực tế: Giao trễ"
        ],
        columns=[
            "Dự đoán: Không trễ",
            "Dự đoán: Giao trễ"
        ]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Teal",
        title="Confusion Matrix"
    )
    fig.update_layout(
        margin=dict(t=45, b=20, l=10, r=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("5. Báo cáo phân loại")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Không giao trễ",
            "Giao trễ"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.subheader("6. Mức độ quan trọng của các đặc trưng")

    trained_preprocessor = model.named_steps["preprocessor"]

    feature_names = (
        trained_preprocessor.get_feature_names_out()
    )

    importances = (
        model.named_steps["classifier"].feature_importances_
    )

    importance_df = pd.DataFrame({
        "Đặc trưng": feature_names,
        "Mức độ quan trọng": importances
    })

    importance_df = importance_df.sort_values(
        by="Mức độ quan trọng",
        ascending=False
    )

    importance_df = importance_df[
        importance_df["Mức độ quan trọng"] > 0
    ]

    fig = px.bar(
        importance_df.head(15),
        x="Mức độ quan trọng",
        y="Đặc trưng",
        orientation="h",
        title="Top 15 đặc trưng quan trọng",
        color="Mức độ quan trọng",
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        margin=dict(t=45, b=20, l=10, r=10),
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("7. Quy tắc cây quyết định")

    tree_classifier = model.named_steps["classifier"]

    display_max_depth = st.slider(
        "Chọn độ sâu hiển thị sơ đồ cây",
        min_value=1,
        max_value=max(1, int(tree_classifier.get_depth())),
        value=min(3, max(1, int(tree_classifier.get_depth()))),
        key="tree_plot_depth"
    )

    fig_tree, ax_tree = plt.subplots(figsize=(20, 9), dpi=150)
    plot_tree(
        tree_classifier,
        max_depth=display_max_depth,
        feature_names=list(feature_names),
        class_names=["Không giao trễ", "Giao trễ"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax_tree
    )
    fig_tree.tight_layout()
    st.pyplot(fig_tree, use_container_width=True)
    plt.close(fig_tree)

    tree_rules = export_text(
        tree_classifier,
        feature_names=list(feature_names)
    )

    st.code(
        tree_rules,
        language="text"
    )

    st.subheader("8. Nhận xét")

    st.write(
        """
        Mô hình Decision Tree được sử dụng để nhận diện các yếu tố
        có liên quan đến nguy cơ giao hàng trễ.

        Các chỉ số Accuracy, Precision, Recall và F1-score giúp
        đánh giá hiệu quả phân loại trên tập kiểm tra.

        Mức độ quan trọng của đặc trưng và các quy tắc phân nhánh
        giúp người dùng hiểu cách mô hình đưa ra dự đoán.
        """
    )


# =========================================================
# 11. CHÂN TRANG
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer-box">
        LOGI AI | Hệ thống phân tích và dự đoán giao hàng trễ
        ứng dụng Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)