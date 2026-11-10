import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Cấu hình trang
st.set_page_config(
    page_title="Dự Đoán Giá Đất TP.HCM - Nhóm 2",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 ỨNG DỤNG DỰ ĐOÁN GIÁ ĐẤT TP. HỒ CHÍ MINH (NHÓM 2)")
st.markdown("""
Ứng dụng triển khai thực tế từ đề tài **Thực hành huấn luyện mô hình AI** dựa trên bộ dữ liệu `Nhom2_GiaDat.xlsx` (gồm `Nhom2_GiaDat_train.xlsx` và `Nhom2_GiaDat_test.xlsx`).
Sử dụng 2 thuật toán: **Linear Regression (Hồi quy tuyến tính)** và **Decision Tree Regression (Cây quyết định hồi quy)**.
""")

# Hàm trích xuất khu vực từ chuỗi Location
def extract_area_name(loc):
    if pd.isna(loc):
        return "Khác"
    parts = [p.strip() for p in str(loc).split(',')]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]

# Hàm tải và tiền xử lý dữ liệu
@st.cache_data
def load_and_clean_data(train_path, test_path):
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    
    for df in [train_df, test_df]:
        df['Khu_Vuc'] = df['Location'].apply(extract_area_name)
    
    # Lọc nhiễu cơ bản (Outliers) để mô hình học chính xác hơn
    def filter_outliers(df):
        cond = (
            (df['Price'] >= 100) & (df['Price'] <= 200000) &
            (df['Area'] >= 15) & (df['Area'] <= 10000)
        )
        return df[cond].copy()
    
    train_clean = filter_outliers(train_df)
    test_clean = filter_outliers(test_df)
    
    # Lấy danh sách Top 25 khu vực phổ biến nhất
    top_khu_vuc = train_clean['Khu_Vuc'].value_counts().nlargest(25).index.tolist()
    train_clean['Khu_Vuc_Clean'] = train_clean['Khu_Vuc'].apply(lambda x: x if x in top_khu_vuc else 'Khác')
    test_clean['Khu_Vuc_Clean'] = test_clean['Khu_Vuc'].apply(lambda x: x if x in top_khu_vuc else 'Khác')
    
    return train_df, test_df, train_clean, test_clean, top_khu_vuc

# Kiểm tra file tồn tại hoặc cho phép tải lên
st.sidebar.header("📂 Cấu hình Dữ liệu & Mô hình")
train_file = "Nhom2_GiaDat_train.xlsx"
test_file = "Nhom2_GiaDat_test.xlsx"

if not (os.path.exists(train_file) and os.path.exists(test_file)):
    st.sidebar.warning("Chưa tìm thấy file mặc định trong thư mục. Vui lòng tải file lên:")
    up_train = st.sidebar.file_uploader("Tải lên Nhom2_GiaDat_train.xlsx", type=["xlsx"])
    up_test = st.sidebar.file_uploader("Tải lên Nhom2_GiaDat_test.xlsx", type=["xlsx"])
    if up_train and up_test:
        train_file, test_file = up_train, up_test
    else:
        st.stop()

train_raw, test_raw, train_clean, test_clean, top_khu_vuc = load_and_clean_data(train_file, test_file)

# Huấn luyện mô hình
@st.cache_resource
def train_models(train_data, test_data):
    num_cols = ['Area', 'Width', 'Length', 'Alley Width']
    cat_cols = ['Position', 'Road Type', 'Direction', 'Agent Role', 'Khu_Vuc_Clean']
    
    # Điền giá trị thiếu (Imputation) dựa trên tập Train
    medians = train_data[num_cols].median()
    
    def prepare_features(df):
        X = pd.DataFrame()
        for col in num_cols:
            X[col] = df[col].fillna(medians[col])
        for col in cat_cols:
            X[col] = df[col].fillna("Chưa rõ").astype(str)
        return X

    X_train_raw = prepare_features(train_data)
    X_test_raw = prepare_features(test_data)
    
    # One-Hot Encoding đồng bộ giữa Train và Test
    combined = pd.concat([X_train_raw, X_test_raw], keys=['train', 'test'])
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    X_train = combined_encoded.xs('train')
    X_test = combined_encoded.xs('test')
    
    # Biến đổi Logarit cho biến mục tiêu Price để giảm độ lệch
    y_train_log = np.log1p(train_data['Price'])
    y_test_actual = test_data['Price'].values
    
    # Thuật toán 1: Linear Regression (Hồi quy tuyến tính)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_log)
    lr_pred = np.expm1(lr_model.predict(X_test))
    lr_pred = np.clip(lr_pred, 0, None)
    
    # Thuật toán 2: Decision Tree Regression (Cây quyết định hồi quy)
    dt_model = DecisionTreeRegressor(max_depth=12, min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train_log)
    dt_pred = np.expm1(dt_model.predict(X_test))
    dt_pred = np.clip(dt_pred, 0, None)
    
    metrics = {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test_actual, lr_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_actual, lr_pred)),
            "R2": r2_score(y_test_actual, lr_pred),
            "Preds": lr_pred
        },
        "Decision Tree Regression": {
            "MAE": mean_absolute_error(y_test_actual, dt_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_actual, dt_pred)),
            "R2": r2_score(y_test_actual, dt_pred),
            "Preds": dt_pred
        }
    }
    
    return lr_model, dt_model, X_train.columns.tolist(), medians, metrics, y_test_actual

with st.spinner("Đang huấn luyện 2 mô hình (Linear Regression & Decision Tree Regression) trên tập Train..."):
    lr_model, dt_model, feature_cols, medians, metrics, y_test_actual = train_models(train_clean, test_clean)

st.sidebar.success(f"Đã tải: {len(train_raw):,} mẫu Train | {len(test_raw):,} mẫu Test")
selected_algo = st.sidebar.selectbox(
    "Chọn thuật toán dự đoán chính:",
    ["Decision Tree Regression", "Linear Regression"]
)

# Giao diện chia làm 3 Tab
tab1, tab2, tab3 = st.tabs(["🔮 Dự đoán Giá đất Mới", "📊 Đánh giá & So sánh 2 Thuật toán", "🗂️ Khám phá Bộ dữ liệu"])

with tab1:
    st.subheader("Nhập thông số lô đất cần định giá")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.number_input("Diện tích đất - Area (m²)", min_value=15.0, max_value=10000.0, value=100.0, step=5.0)
        width = st.number_input("Chiều ngang mặt tiền - Width (m)", min_value=1.0, max_value=200.0, value=5.0, step=0.5)
        length = st.number_input("Chiều dài - Length (m)", min_value=1.0, max_value=500.0, value=20.0, step=0.5)
        
    with col2:
        khu_vuc = st.selectbox("Khu vực (Quận/Huyện/Phường)", options=top_khu_vuc + ["Khác"])
        position = st.selectbox("Vị trí lô đất (Position)", options=["Đường chính", "Trong hẻm", "Chưa rõ"])
        alley_width = st.number_input("Độ rộng hẻm/đường trước đất (m)", min_value=1.0, max_value=60.0, value=6.0, step=0.5)
        
    with col3:
        road_type = st.selectbox("Loại đường (Road Type)", options=["Đường nhựa", "Đường bê tông", "Đường đá", "Đường đất", "Chưa rõ"])
        direction = st.selectbox("Hướng đất (Direction)", options=["Nam", "Đông Nam", "Bắc", "Đông", "Tây", "Tây Nam", "Đông Bắc", "Tây Bắc", "Chưa rõ"])
        agent_role = st.selectbox("Người đăng tin (Agent Role)", options=["Chính chủ", "Môi giới"])

    if st.button("🚀 Dự đoán Giá Đất ngay", use_container_width=True):
        # Tạo dataframe 1 dòng cho dữ liệu mới
        input_dict = {
            'Area': area,
            'Width': width,
            'Length': length,
            'Alley Width': alley_width,
            'Position': position,
            'Road Type': road_type,
            'Direction': direction,
            'Agent Role': agent_role,
            'Khu_Vuc_Clean': khu_vuc
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        
        # Căn chỉnh các cột khớp 100% với tập Train
        input_aligned = input_encoded.reindex(columns=feature_cols, fill_value=0)
        
        # Dự đoán bằng cả 2 thuật toán
        pred_lr_val = np.expm1(lr_model.predict(input_aligned)[0])
        pred_dt_val = np.expm1(dt_model.predict(input_aligned)[0])
        
        main_pred = pred_dt_val if selected_algo == "Decision Tree Regression" else pred_lr_val
        other_pred = pred_lr_val if selected_algo == "Decision Tree Regression" else pred_dt_val
        other_name = "Linear Regression" if selected_algo == "Decision Tree Regression" else "Decision Tree Regression"
        don_gia_m2 = main_pred / area
        
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(
            label=f"💰 Giá dự đoán ({selected_algo})",
            value=f"{main_pred:,.0f} Triệu VNĐ",
            delta=f"≈ {main_pred/1000:,.2f} Tỷ VNĐ"
        )
        res_col2.metric(
            label="📐 Đơn giá ước tính / m²",
            value=f"{don_gia_m2:,.1f} Triệu/m²"
        )
        res_col3.metric(
            label="🔄 Đối chiếu thuật toán còn lại",
            value=f"{other_pred/1000:,.2f} Tỷ VNĐ",
            delta=other_name
        )

with tab2:
    st.subheader("Kết quả thực nghiệm trên tập kiểm thử (Nhom2_GiaDat_test.xlsx)")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.info("### 📈 Thuật toán 1: Linear Regression (Hồi quy tuyến tính)")
        st.write(f"- **MAE (Sai số tuyệt đối trung bình):** `{metrics['Linear Regression']['MAE']:,.2f}` Triệu VNĐ")
        st.write(f"- **RMSE (Sai số bình phương trung bình căn):** `{metrics['Linear Regression']['RMSE']:,.2f}` Triệu VNĐ")
        st.write(f"- **R² Score (Độ phù hợp):** `{metrics['Linear Regression']['R2']:.4f}`")
        
    with m_col2:
        st.success("### 🌳 Thuật toán 2: Decision Tree Regression (Cây quyết định hồi quy)")
        st.write(f"- **MAE (Sai số tuyệt đối trung bình):** `{metrics['Decision Tree Regression']['MAE']:,.2f}` Triệu VNĐ")
        st.write(f"- **RMSE (Sai số bình phương trung bình căn):** `{metrics['Decision Tree Regression']['RMSE']:,.2f}` Triệu VNĐ")
        st.write(f"- **R² Score (Độ phù hợp):** `{metrics['Decision Tree Regression']['R2']:.4f}`")
        
    st.markdown("#### 📋 Bảng đối chiếu Giá trị thực tế và Giá trị dự đoán (15 mẫu đầu tiên trong tập Test)")
    compare_df = pd.DataFrame({
        "STT": range(1, 16),
        "Khu vực": test_clean['Khu_Vuc'].iloc[:15].values,
        "Diện tích (m²)": test_clean['Area'].iloc[:15].values,
        "Giá trị thực (Triệu VNĐ)": y_test_actual[:15].round(1),
        "Thuật toán 1 - Linear Regression (Triệu VNĐ)": metrics['Linear Regression']['Preds'][:15].round(1),
        "Thuật toán 2 - Decision Tree Regression (Triệu VNĐ)": metrics['Decision Tree Regression']['Preds'][:15].round(1)
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Thông tin tổng quan Bộ dữ liệu (Nhom2_GiaDat)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng mẫu Đất nền", f"{len(train_raw) + len(test_raw):,}")
    c2.metric("Tập Huấn luyện (Train 80%)", f"{len(train_raw):,}")
    c3.metric("Tập Kiểm thử (Test 20%)", f"{len(test_raw):,}")
    c4.metric("Số thuộc tính (Columns)", f"{train_raw.shape[1]}")
    
    st.markdown("#### 10 dòng dữ liệu đầu tiên trong tập Huấn luyện (`Nhom2_GiaDat_train.xlsx`)")
    st.dataframe(train_raw.head(10), use_container_width=True)
