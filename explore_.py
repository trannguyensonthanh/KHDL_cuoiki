# ==============================================================================
# FILE: explore.py
# CHỨC NĂNG: TRỰC QUAN HÓA DỮ LIỆU XE HƠI (ADVANCED DATA VISUALIZATION)
# DỰA TRÊN: Chapter 3 - Data Visualization & Project Context
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import warnings

# Tắt cảnh báo để output sạch đẹp
warnings.filterwarnings('ignore')

# Cấu hình hiển thị Matplotlib (Font chữ & Style)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 12
# Nếu bị lỗi font tiếng Việt, bạn có thể cần set font cụ thể (vd: Arial/Roboto) tùy OS
# plt.rcParams['font.family'] = 'sans-serif' 

class CarDataVisualizer:
    def __init__(self, file_path="D:\\Download\\learningdocument\\Khoa học dữ liệu\\cuoiki\\KHDL\\scraped_cars.csv"):
        self.file_path = file_path
        self.df = self.load_and_clean_data()

    def load_and_clean_data(self):
        """
        Đọc và làm sạch dữ liệu thô (Raw CSV -> Clean DataFrame)
        Áp dụng logic tương tự recommender_engine.py để đảm bảo tính nhất quán.
        """
        print(f"🔄 Đang đọc và xử lý dữ liệu từ {self.file_path}...")
        try:
            df = pd.read_csv(self.file_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(self.file_path, encoding='utf-8')

        # 1. Xử lý Giá (Price) - Chuyển về đơn vị Tỷ VNĐ cho dễ nhìn
        # Dữ liệu gốc có thể là số nguyên lớn, ta chia cho 1 tỷ
        df['price_billion'] = pd.to_numeric(df['price'], errors='coerce').fillna(0) / 1_000_000_000
        
        # Lọc bỏ xe giá quá ảo (vd: 0đ hoặc > 50 tỷ - outlier) để biểu đồ đẹp hơn
        df = df[(df['price_billion'] > 0.1) & (df['price_billion'] < 20)]

        # 2. Xử lý Hãng (Brand)
        df['brand'] = df['brand'].astype(str).str.strip().str.title()
        
        # 3. Xử lý Mã lực (Horsepower)
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        # Fill mean cho giá trị thiếu để không mất dữ liệu khi vẽ Scatter
        df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)

        # 4. Xử lý Năm (Year) & Tuổi xe (Age)
        current_year = datetime.now().year
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(current_year)
        df['age'] = current_year - df['year']

        # 5. Phân nhóm Nhiên liệu (Fuel)
        def clean_fuel(f):
            f = str(f).lower()
            if 'điện' in f or 'electric' in f: return 'Electric'
            if 'hybrid' in f: return 'Hybrid'
            if 'dầu' in f or 'diesel' in f: return 'Diesel'
            return 'Petrol'
        df['fuel_group'] = df['fuelType'].apply(clean_fuel)

        # 6. Phân nhóm Xe (Dựa trên logic của Engine)
        def classify_type(row):
            text = (str(row['name']) + " " + str(row.get('description', ''))).lower()
            seats = pd.to_numeric(row['seats'], errors='coerce')
            if pd.isna(seats): seats = 5
            
            if re.search(r'bán tải|pickup|ranger|triton', text): return 'Pickup'
            if re.search(r'suv|cross|gầm cao|cx-|cr-v|tucson', text): return 'SUV/CUV'
            if seats >= 7: return 'MPV/7-Seat'
            if re.search(r'hatchback|yaris|swift|morning', text): return 'Hatchback'
            return 'Sedan'
        
        df['body_type'] = df.apply(classify_type, axis=1)

        print(f"✅ Đã xử lý xong: {len(df)} dòng dữ liệu sạch.")
        return df

    # ==========================================================================
    # DASHBOARD 1: TỔNG QUAN THỊ TRƯỜNG (Market Overview)
    # Bao gồm: Histogram (Giá), Bar Chart (Hãng), Pie Chart (Hộp số/Nhiên liệu)
    # ==========================================================================
    def plot_market_overview(self):
        df = self.df
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        plt.suptitle('DASHBOARD 1: TỔNG QUAN THỊ TRƯỜNG XE Ô TÔ', fontsize=24, weight='bold', color='#333')

        # 1. Top 10 Hãng xe có giá trung bình rẻ nhất (Bar Chart - Slide 13)
        avg_price_by_brand = df.groupby('brand')['price_billion'].mean().nsmallest(10)
        sns.barplot(x=avg_price_by_brand.values, y=avg_price_by_brand.index, ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Top 10 Hãng Xe Có Giá Trung Bình Rẻ Nhất', fontsize=16)
        axes[0, 0].set_xlabel('Giá trung bình (Tỷ VNĐ)')
        # Add labels
        for i, v in enumerate(avg_price_by_brand.values):
            axes[0, 0].text(v + 0.05, i, f'{v:.2f}', color='black', va='center')

        # 2. Phân phối Giá xe (Histogram & KDE - Slide 39, 42)
        # Sử dụng log scale hoặc giới hạn để nhìn rõ hơn
        sns.histplot(df['price_billion'], bins=30, kde=True, ax=axes[0, 1], color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Phân Phối Giá Xe (Tỷ VNĐ)', fontsize=16)
        axes[0, 1].set_xlabel('Giá (Tỷ VNĐ)')
        axes[0, 1].set_ylabel('Tần suất')
        # Vẽ đường trung bình
        mean_price = df['price_billion'].mean()
        axes[0, 1].axvline(mean_price, color='red', linestyle='--', label=f'TB: {mean_price:.2f} Tỷ')
        axes[0, 1].legend()

        # 3. Cơ cấu Nhiên liệu (Donut Chart)
        fuel_counts = df['fuel_group'].value_counts()
        axes[1, 0].pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', startangle=90, 
                       colors=sns.color_palette('pastel'), wedgeprops={'width': 0.4})
        axes[1, 0].set_title('Tỷ Lệ Các Loại Nhiên Liệu', fontsize=16)

        # 4. Phân loại Kiểu dáng xe (Countplot - Slide 16)
        sns.countplot(x='body_type', data=df, ax=axes[1, 1], palette='magma', order=df['body_type'].value_counts().index)
        axes[1, 1].set_title('Số Lượng Xe Theo Kiểu Dáng (Body Type)', fontsize=16)
        axes[1, 1].set_xlabel('')
        axes[1, 1].set_ylabel('Số lượng')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0, w_pad=3.0)
        plt.show()

    # ==========================================================================
    # DASHBOARD 2: PHÂN TÍCH CHUYÊN SÂU & TƯƠNG QUAN (Deep Dive & Correlation)
    # Bao gồm: Scatter Plot, Box Plot, Heatmap
    # ==========================================================================
    def plot_deep_analysis(self):
        df = self.df
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        plt.suptitle('DASHBOARD 2: PHÂN TÍCH TƯƠNG QUAN & PHÂN KHÚC', fontsize=24, weight='bold', color='#333')

        # 1. Tương quan Mã lực vs Giá xe (Scatter Plot - Slide 18-22)
        # Màu sắc (hue) thể hiện loại nhiên liệu
        sns.scatterplot(x='horsepower', y='price_billion', data=df, hue='fuel_group', 
                        style='body_type', alpha=0.7, s=100, ax=axes[0, 0], palette='deep')
        axes[0, 0].set_title('Mã Lực (HP) vs Giá Xe (Có phân loại nhiên liệu)', fontsize=16)
        axes[0, 0].set_xlabel('Mã lực (Horsepower)')
        axes[0, 0].set_ylabel('Giá (Tỷ VNĐ)')
        
        # Annotation (Slide 49-50): Chỉ ra xe mạnh nhất/đắt nhất
        max_hp_row = df.loc[df['horsepower'].idxmax()]
        axes[0, 0].annotate(f"Max HP: {max_hp_row['name']}", 
                            xy=(max_hp_row['horsepower'], max_hp_row['price_billion']),
                            xytext=(max_hp_row['horsepower']-100, max_hp_row['price_billion']+2),
                            arrowprops=dict(facecolor='black', shrink=0.05))

        # 2. Phân bố giá theo Hãng xe (Box Plot - Slide 25-27)
        # Chỉ lấy Top 8 hãng để biểu đồ thoáng
        top_8_brands = df['brand'].value_counts().nlargest(8).index
        df_top8 = df[df['brand'].isin(top_8_brands)]
        
        sns.boxplot(x='brand', y='price_billion', data=df_top8, ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Biên Độ Giá Của Top 8 Hãng Xe (Box Plot)', fontsize=16)
        axes[0, 1].set_xlabel('Hãng xe')
        axes[0, 1].set_ylabel('Giá (Tỷ VNĐ)')

        # 3. Tương quan Giá theo Năm sản xuất (Line Plot/Reg Plot - Slide 8)
        # Xem xu hướng mất giá của xe
        sns.regplot(x='year', y='price_billion', data=df, ax=axes[1, 0], 
                    scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        axes[1, 0].set_title('Xu Hướng Giá Xe Theo Năm Sản Xuất', fontsize=16)
        axes[1, 0].set_xlabel('Năm sản xuất')
        axes[1, 0].set_ylabel('Giá (Tỷ VNĐ)')

        # 4. Ma trận tương quan (Heatmap - Slide 34-37 về Density/Contour nhưng áp dụng Heatmap cho Correlation)
        # Chọn các cột số
        numeric_cols = df[['price_billion', 'year', 'horsepower', 'seats', 'age']]
        corr_matrix = numeric_cols.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1, 1])
        axes[1, 1].set_title('Ma Trận Tương Quan Giữa Các Thông Số', fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0, w_pad=3.0)
        plt.show()

    # ==========================================================================
    # DASHBOARD 3: XU HƯỚNG NÂNG CAO (Advanced Trends)
    # Bao gồm: Violin Plot, Multi-line Plot
    # ==========================================================================
    def plot_advanced_trends(self):
        df = self.df
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plt.suptitle('DASHBOARD 3: PHÂN TÍCH NÂNG CAO', fontsize=24, weight='bold', color='#333')

        # 1. Violin Plot: Giá theo Kiểu dáng (Kết hợp Boxplot và Density - Slide 42)
        sns.violinplot(x='body_type', y='price_billion', data=df, ax=axes[0], palette='muted')
        axes[0].set_title('Mật Độ Giá Theo Kiểu Dáng Xe (Violin Plot)', fontsize=16)
        axes[0].set_ylabel('Giá (Tỷ VNĐ)')

        # 2. Giá trung bình theo Năm của từng Hãng (Multi-line Plot)
        # Chọn top 5 hãng để vẽ
        top_5_brands = df['brand'].value_counts().nlargest(5).index
        df_trend = df[df['brand'].isin(top_5_brands)]
        
        # Group by Year and Brand
        trend_data = df_trend.groupby(['year', 'brand'])['price_billion'].mean().reset_index()
        # Chỉ lấy dữ liệu từ năm 2010 trở lại đây cho đỡ nhiễu
        trend_data = trend_data[trend_data['year'] >= 2010]

        sns.lineplot(x='year', y='price_billion', hue='brand', data=trend_data, marker='o', ax=axes[1], linewidth=2.5)
        axes[1].set_title('Biến Động Giá Trung Bình Các Hãng Theo Năm', fontsize=16)
        axes[1].set_ylabel('Giá TB (Tỷ VNĐ)')
        axes[1].grid(True, linestyle='--')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Khởi động trình trực quan hóa dữ liệu (Ultimate Car Viz)...")
    
    # Khởi tạo Visualizer
    viz = CarDataVisualizer()
    
    # 1. Vẽ Dashboard Tổng quan
    viz.plot_market_overview()
    
    # 2. Vẽ Dashboard Phân tích sâu
    viz.plot_deep_analysis()
    
    # 3. Vẽ Dashboard Xu hướng nâng cao
    viz.plot_advanced_trends()
    
    print("✅ Đã hoàn tất vẽ biểu đồ.")