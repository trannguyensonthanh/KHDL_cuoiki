# ==============================================================================
# FILE: recommender_engine.py
# CHỨC NĂNG: Bộ não xử lý chính (Data -> Model -> Recommend -> Evaluate)
# ==============================================================================
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
import os
import re
import pickle
import numpy as np
import pandas as pd
from math import sqrt
import torch.nn.functional as F
from config import GEMINI_API_KEY
# Thư viện AI & Machine Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset as SurpriseDataset, Reader, SVD, dump
import json
import time
from google import genai
from google.genai import types
from datetime import datetime
# Cấu hình thiết bị (Ưu tiên GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Thiết bị tính toán: {device}")



# ==============================================================================
# 1. XỬ LÝ DỮ LIỆU (DATA PROCESSOR)
# ==============================================================================
class CarDataProcessor:
    def __init__(self, file_path="scraped_cars.csv"):
        self.file_path = file_path

    def process(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file {self.file_path}")

        print(f"1. [Data] Đọc dữ liệu từ CSV: {self.file_path}")
        try:
            df = pd.read_csv(self.file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(self.file_path, encoding='utf-8')

        # ======================================================================
        # 1. CHUẨN HÓA SỐ LIỆU CƠ BẢN (NUMERIC CLEANING)
        # ======================================================================
        df['id'] = df['id'].astype(str) 
        # Giá xe (VNĐ)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        
        # Năm sản xuất
        current_year = datetime.now().year
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(current_year - 5)
        df['age'] = current_year - df['year'] # Tuổi xe
        
        # Mã lực (Power)
        df['power'] = pd.to_numeric(df['horsepower'], errors='coerce').fillna(df['horsepower'].mean())
        
        # Số chỗ ngồi
        df['n_seats'] = pd.to_numeric(df['seats'], errors='coerce').fillna(5)

        # ODO (Số km đã đi) - Giả lập thông minh nếu thiếu
        # Logic: Xe lướt (<2 tuổi) đi ít (10k/năm), xe cũ đi nhiều (15k/năm)
        if 'odo' not in df.columns:
            df['mileage'] = df.apply(
                lambda x: (current_year - x['year']) * (10000 if (current_year - x['year']) < 3 else 15000), 
                axis=1
            )
        else:
            # Nếu có cột odo thật thì dùng, clean chữ 'km'
            df['mileage'] = df['odo'].astype(str).str.replace(r'\D', '', regex=True)
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce').fillna((current_year - df['year']) * 12000)

        # ======================================================================
        # 2. PHÂN LOẠI DÒNG XE THÔNG MINH (ADVANCED BODY TYPE CLASSIFICATION)
        # ======================================================================
        def classify_car_type(row):
            text = (str(row['name']) + " " + str(row.get('description', ''))).lower()
            seats = row['n_seats']
            
            # Ưu tiên theo từ khóa
            if re.search(r'bán tải|pickup|ranger|triton|hilux|navara|bt-50', text):
                return 'pickup'
            if re.search(r'mpv|carnival|stargazer|custin|innova|xpander|veloz|avanza|xl7|ertiga', text):
                return 'mpv'
            if re.search(r'suv|cross|gầm cao|cx-|cr-v|tucson|santafe|sorento|everest|fortuner|glc|x3|x5', text):
                return 'suv'
            if re.search(r'hatchback|yaris|swift|morning|i10|wigo|fadil|jazz', text):
                return 'hatchback'
            if re.search(r'coupe|mui trần|convertible|sport|2 cửa', text):
                return 'sport'
            
            # Fallback theo số chỗ
            if seats >= 7: return 'mpv' # 7 chỗ thường là MPV hoặc SUV (đã lọc ở trên) -> gán MPV cho chắc
            return 'sedan' # Mặc định còn lại là Sedan (Vios, Accent, Camry...)

        df['car_type'] = df.apply(classify_car_type, axis=1)

        # ======================================================================
        # 3. CHUẨN HÓA TEXT (CATEGORY NORMALIZATION)
        # ======================================================================
        
        # Hãng xe
        df['make'] = df['brand'].astype(str).str.strip().str.title()
        
        # Nhiên liệu (Gộp nhóm)
        def clean_fuel(f):
            f = str(f).lower()
            if 'điện' in f or 'electric' in f: return 'Electric'
            if 'hybrid' in f: return 'Hybrid'
            if 'dầu' in f or 'diesel' in f: return 'Diesel'
            return 'Petrol'
        df['fuel_category'] = df['fuelType'].apply(clean_fuel)

        # Hộp số
        df['is_automatic'] = df['transmission'].astype(str).apply(
            lambda x: 1 if 'tự động' in x.lower() or 'at' in x.lower() else 0
        )

        # ======================================================================
        # 4. TRÍCH XUẤT TÍNH NĂNG CAO CẤP (FEATURE EXTRACTION)
        # ======================================================================
        # Tạo cột điểm công nghệ (Tech Score) để phân biệt bản thiếu/đủ
        
        # Danh sách từ khóa tính năng xịn
        tech_keywords = {
            'has_sunroof': ['cửa sổ trời', 'sunroof', 'panorama'],
            'has_adas': ['adas', 'giữ làn', 'phanh tự động', 'cảnh báo va chạm', 'honda sensing', 'toyota safety sense'],
            'has_360': ['camera 360', 'cam 360'],
            'has_leather': ['ghế da', 'da nappa'],
            'has_smartkey': ['start/stop', 'khởi động nút bấm', 'smartkey'],
            'has_cruise': ['cruise control', 'ga tự động']
        }

        # Tạo các cột flag (0/1)
        full_text = (df['features'].fillna('') + " " + df['description'].fillna('')).str.lower()
        
        for col, keywords in tech_keywords.items():
            pattern = "|".join(keywords)
            df[col] = full_text.str.contains(pattern, regex=True).astype(int)

        # Tổng hợp thành Tech Score (0 -> 10)
        df['tech_score'] = (
            df['has_sunroof'] * 1.5 + 
            df['has_adas'] * 2.0 + 
            df['has_360'] * 1.5 + 
            df['has_leather'] * 1.0 + 
            df['has_smartkey'] * 1.0 +
            df['has_cruise'] * 1.0
        )

        # ======================================================================
        # 5. PHÂN KHÚC & LOGIC NGHIỆP VỤ (BUSINESS LOGIC)
        # ======================================================================
        
        # Price Code (Phân khúc giá chi tiết hơn cho VN)
        # 1: <400tr (Xe cỏ)
        # 2: 400-700tr (Phổ thông)
        # 3: 700-1.2 tỷ (Trung cấp/SUV C)
        # 4: 1.2-2.5 tỷ (Cận sang/Sang nhỏ)
        # 5: >2.5 tỷ (Xe sang/Siêu sang)
        def get_price_code(p):
            if p < 400_000_000: return 1
            if p < 700_000_000: return 2
            if p < 1_200_000_000: return 3
            if p < 2_500_000_000: return 4
            return 5
        df['price_code'] = df['price'].apply(get_price_code)

        # Cờ phân loại (Dùng cho Hard Filter và Persona Generator)
        df['is_family'] = ((df['n_seats'] >= 5) & (df['car_type'].isin(['suv', 'mpv', 'sedan']))).astype(int)
        df['is_service'] = ((df['price_code'] <= 2) & (df['fuel_category'].isin(['Diesel', 'Petrol'])) & (df['n_seats'] >= 4)).astype(int)
        df['is_luxury'] = ((df['price_code'] >= 4) | (df['make'].isin(['Mercedes-Benz', 'Bmw', 'Audi', 'Lexus', 'Porsche', 'Land-Rover', 'Volvo']))).astype(int)
        df['is_sport'] = ((df['power'] > 250) | (df['car_type'] == 'sport')).astype(int)
        df['is_green'] = (df['fuel_category'].isin(['Electric', 'Hybrid'])).astype(int)

        print(f"   -> Đã xử lý xong {len(df)} dòng dữ liệu xe.")
        print(f"   -> Các cột mới: car_type, tech_score, is_green, fuel_category...")
        
        return df
# ==============================================================================
# 2. SINH DỮ LIỆU GIẢ LẬP (PERSONA GENERATOR)
# ==============================================================================
class PersonaGenerator:
    """
    Sinh dữ liệu User giả lập bằng cách dùng LLM (Gemma-3) đóng vai người dùng thật.
    Chiến lược: "Prototype & Clone" (Tạo mẫu bằng AI -> Nhân bản bằng Toán học).
    """
    def __init__(self, df_cars, num_users=1000):
        self.df_cars = df_cars
        self.num_users = num_users
        
        # Khởi tạo Client Gemini
        try:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.model_name = "gemini-2.5-flash-lite" # Hoặc "gemini-2.5-flash" nếu account bạn có quyền
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Gemini: {e}. Vui lòng kiểm tra API Key.")
            self.client = None

    def _get_llm_ratings_for_persona(self, persona_name, persona_desc, car_samples):
        """
        Gửi danh sách xe cho LLM và yêu cầu chấm điểm theo vai (Persona)
        """
        print(f"   🤖 AI đang đóng vai '{persona_name}' để chấm điểm xe...")
        
        # Tạo text mô tả danh sách xe rút gọn để tiết kiệm token
        cars_text = ""
        for _, row in car_samples.iterrows():
            # Gom thông tin quan trọng để AI đánh giá
            cars_text += (f"- ID: {row['id']} | Xe: {row['name']} | Hãng: {row['brand']} | "
                          f"Giá: {row['price']:,} VNĐ | Loại: {row['seats']} chỗ, {row['fuelType']}, {row.get('power', 100)}HP\n")

        prompt = f"""
        Bạn hãy nhập vai một người dùng ô tô với hồ sơ sau:
        "{persona_desc}"

        Dưới đây là danh sách các mẫu xe ô tô thực tế:
        {cars_text}

        Nhiệm vụ:
        1. Dựa trên tính cách và nhu cầu của bạn, hãy chấm điểm từng chiếc xe trên thang điểm từ 1.0 đến 5.0.
        2. Hãy chấm điểm công tâm dựa trên kiến thức thực tế (Ví dụ: Xe sang thì đắt nhưng sướng, xe cỏ thì bền nhưng ồn, xe điện thì hiện đại...).
        3. TRẢ VỀ KẾT QUẢ DẠNG JSON THUẦN (Array of Objects), không giải thích gì thêm.
        
        Format mẫu:
        [
            {{"id": "ID_XE_1", "rating": 4.5}},
            {{"id": "ID_XE_2", "rating": 2.0}}
        ]
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7) # Temp cao chút cho sáng tạo
            )
            # Clean JSON string (phòng trường hợp LLM thêm markdown)
            json_str = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(json_str)
        except Exception as e:
            print(f"   ⚠️ Lỗi khi gọi AI cho {persona_name}: {e}")
            return []


    def generate_ratings(self):
        # Nếu không có API Key hoặc Client lỗi -> Fallback về logic if-else cũ (Code an toàn)
        if not self.client:
            print("⚠️ Không có kết nối AI. Chuyển về chế độ sinh dữ liệu thủ công (Rule-based).")
            return self._generate_ratings_fallback()

        print("2. [Data] Đang sinh Ratings thông minh bằng AI (Gemma-2/Flash)...")
        
        all_ratings = []
        
        # 1. Chọn mẫu xe (Sampling)
        # Không thể gửi cả 1000 xe cho AI (tốn tiền/token). 
        # Ta chọn 50 xe tiêu biểu đại diện cho các phân khúc.
        if len(self.df_cars) > 50:
            sample_cars = self.df_cars.sample(n=50, random_state=42)
        else:
            sample_cars = self.df_cars

        # 2. Định nghĩa Persona chi tiết (Prompt Engineering)
        personas = {
            'Student': "Tôi là sinh viên mới ra trường, thu nhập thấp. Tôi cần xe giá rẻ, tiết kiệm xăng, bền bỉ, ít hỏng vặt (như Vios, Morning). Tôi ghét xe sang vì nuôi tốn kém.",
            'Family': "Tôi là người đàn ông của gia đình. Tôi ưu tiên xe rộng rãi (5-7 chỗ), an toàn, gầm cao (SUV/MPV) để chở vợ con đi chơi. Giá cả hợp lý là được.",
            'Boss': "Tôi là doanh nhân thành đạt. Tôi cần xe sang trọng, thương hiệu lớn (Mercedes, BMW, Lexus, Porsche) để thể hiện đẳng cấp. Giá cả không quan trọng, miễn là tiện nghi và êm ái.",
            'Racer': "Tôi đam mê tốc độ và công nghệ. Tôi thích xe có động cơ mạnh mẽ (mã lực cao), thiết kế thể thao hoặc xe điện công nghệ cao. Tôi không thích xe yếu ớt."
        }

        # 3. Vòng lặp sinh dữ liệu
        users_per_persona = self.num_users // len(personas) # Chia đều user cho mỗi nhóm

        for p_name, p_desc in personas.items():
            # A. Lấy điểm gốc từ AI (Prototype Ratings)
            base_ratings = self._get_llm_ratings_for_persona(p_name, p_desc, sample_cars)
            
            if not base_ratings: continue # Skip nếu lỗi

            # B. Nhân bản ra nhiều User (Cloning with Noise)
            print(f"   -> Đang nhân bản {users_per_persona} user cho nhóm {p_name}...")
            
            for _ in range(users_per_persona):
                # Tạo ID user ngẫu nhiên
                user_id = np.random.randint(100000, 999999)
                
                for item in base_ratings:
                    car_id = str(item.get('id'))
                    base_score = float(item.get('rating', 3.0))
                    
                    # THÊM NHIỄU (Noise): Để các user không giống nhau 100%
                    # Normal distribution: mean=0, std=0.4 (dao động khoảng +/- 0.8 điểm)
                    noise = np.random.normal(0, 0.4)
                    final_score = np.clip(base_score + noise, 1, 5)
                    
                    # Random drop: User không nhất thiết phải rate hết tất cả xe mẫu
                    # Giả sử user chỉ rate 70% số xe mẫu
                    if np.random.rand() < 0.7:
                        all_ratings.append({
                            'user_id': user_id,
                            'car_id': car_id,
                            'rating': round(final_score, 1), # Làm tròn 1 số lẻ
                            'persona': p_name
                        })
            
            # Nghỉ 1 chút để tránh hit rate limit của Google
            time.sleep(2)

        df_ratings = pd.DataFrame(all_ratings)
        print(f"✅ Đã sinh xong {len(df_ratings)} ratings từ AI.")
        return df_ratings

    def _generate_ratings_fallback(self):
        print("2. [Data] Đang sinh Ratings giả lập theo Persona (Student, Family, Boss)...")
        ratings = []
        
        # Định nghĩa các nhóm người dùng
        personas = ['Student', 'Family', 'Boss', 'Racer']
        
        for uid in range(self.num_users):
            p = np.random.choice(personas)
            
            # Mỗi user đánh giá ngẫu nhiên 15-20 xe
            sample_cars = self.df_cars.sample(n=np.random.randint(15, 20))
            
            for _, car in sample_cars.iterrows():
                base_score = 3.0
                
                # --- LOGIC GIẢ LẬP SỞ THÍCH ---
                if p == 'Student': # Thích rẻ, ghét đắt
                    if car['is_cheap']: base_score += 2.0
                    if car['is_luxury']: base_score -= 1.0
                    if car['mileage'] > 200000: base_score -= 0.5
                
                elif p == 'Family': # Thích rộng, an toàn
                    if car['is_family']: base_score += 2.0
                    if car['car_type'] in ['coupe', 'convertible']: base_score -= 1.0
                    if car['year'] > 2016: base_score += 0.5
                
                elif p == 'Boss': # Thích sang, ghét rẻ
                    if car['is_luxury']: base_score += 2.0
                    if car['is_cheap']: base_score -= 1.0
                    if car['make'] in ['Mercedes-Benz', 'Bmw', 'Audi', 'Lexus']: base_score += 1.0
                
                elif p == 'Racer': # Thích mạnh
                    if car['power'] > 150: base_score += 2.0
                    if car['car_type'] in ['coupe', 'convertible']: base_score += 1.0

                # Thêm nhiễu (Noise) để dữ liệu tự nhiên hơn
                final_score = np.clip(base_score + np.random.uniform(-0.5, 0.5), 1, 5)
                
                ratings.append({
                    'user_id': uid, 
                    'car_id': car['id'], 
                    'rating': final_score, 
                    'persona': p
                })
        
        return pd.DataFrame(ratings)
# ==============================================================================
# 3. MODEL: TWO-TOWER NEURAL NETWORK (PYTORCH)
# ==============================================================================
class AdvancedTwoTowerNet(nn.Module):
    def __init__(self, 
                 n_users, n_personas, 
                 n_items, n_brands, n_car_types, 
                 embedding_dim=32):
        super(AdvancedTwoTowerNet, self).__init__()
        
        # --- USER TOWER ---
        # 1. Embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.persona_emb = nn.Embedding(n_personas, 8) # Persona ít nên dim nhỏ
        
        # 2. Dense Layers (MLP)
        # Input size = User Emb + Persona Emb
        self.user_layers = nn.Sequential(
            nn.Linear(embedding_dim + 8, 128),
            nn.BatchNorm1d(128), # Giúp train ổn định hơn
            nn.ReLU(),
            nn.Dropout(0.3),     # Chống overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)    # Output vector size 32
        )
        
        # --- ITEM TOWER (XE) ---
        # 1. Embeddings (Cho dữ liệu phân loại)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.brand_emb = nn.Embedding(n_brands, 16)
        self.type_emb = nn.Embedding(n_car_types, 8)
        
        # 2. Feature Transformation (Cho dữ liệu số: Price, Year, Power...)
        # Input: 5 chỉ số số học (Price norm, Year norm, Power norm, Seats norm, Tech Score)
        self.numeric_trans = nn.Linear(5, 16) 
        
        # 3. Dense Layers (MLP)
        # Input size = Item(32) + Brand(16) + Type(8) + Numeric(16) = 72
        self.item_layers = nn.Sequential(
            nn.Linear(embedding_dim + 16 + 8 + 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)    # Output vector size 32 (Phải khớp User Tower)
        )

    def forward(self, user_inputs, item_inputs):
        """
        user_inputs: [u_idx, persona_idx]
        item_inputs: [i_idx, brand_idx, type_idx, price, year, power, seats, tech]
        """
        
        # --- USER TOWER FORWARD ---
        u_idx = user_inputs[:, 0].long()
        p_idx = user_inputs[:, 1].long()
        
        u_vec = self.user_emb(u_idx)
        p_vec = self.persona_emb(p_idx)
        
        # Nối vector (User ID + Persona)
        user_combined = torch.cat([u_vec, p_vec], dim=1)
        user_rep = self.user_layers(user_combined)
        
        # --- ITEM TOWER FORWARD ---
        i_idx = item_inputs[:, 0].long()
        b_idx = item_inputs[:, 1].long()
        t_idx = item_inputs[:, 2].long()
        # Các chỉ số số học (Price, Year...)
        numerics = item_inputs[:, 3:].float() 
        
        i_vec = self.item_emb(i_idx)
        b_vec = self.brand_emb(b_idx)
        t_vec = self.type_emb(t_idx)
        n_vec = F.relu(self.numeric_trans(numerics)) # Transform số học
        
        # Nối tất cả đặc trưng xe lại
        item_combined = torch.cat([i_vec, b_vec, t_vec, n_vec], dim=1)
        item_rep = self.item_layers(item_combined)
        
        # --- OUTPUT: DOT PRODUCT ---
        # Tính tương đồng giữa Vector User tổng hợp và Vector Xe tổng hợp
        return (user_rep * item_rep).sum(dim=1)

# ==============================================================================
# 4. HỆ THỐNG GỢI Ý CHÍNH (MAIN CLASS)
# ==============================================================================
class CarRecommendationSystem:
    def __init__(self, csv_path="D:\\Download\\learningdocument\\Khoa học dữ liệu\\cuoiki\\KHDL\\scraped_cars.csv"):
        self.cp_dir = "checkpoints"
        if not os.path.exists(self.cp_dir): os.makedirs(self.cp_dir)

        # 1. Load & Process Data
        self.processor = CarDataProcessor(csv_path)
        self.df_cars = self.processor.process()

        # 2. Prepare Ratings & Encoders
        self._prepare_data()
        # --- NẠP FEEDBACK ---
        self.load_feedback_data() 
        # 3. Train or Load Models
        self._load_or_train_models()

        # 4. Build Item-Item Similarity Matrix (Slide Knowledge)
        self._build_item_similarity()

    def _prepare_data(self):
        path_ratings = f"{self.cp_dir}/ratings_gen.csv"
        
        if os.path.exists(path_ratings):
            print("2. [Data] Load ratings đã lưu từ cache...")
            self.df_ratings = pd.read_csv(path_ratings)
        else:
            gen = PersonaGenerator(self.df_cars)
            self.df_ratings = gen.generate_ratings()
            self.df_ratings.to_csv(path_ratings, index=False)
        # Ép cả 2 về string để tránh lỗi "object and int64"
        self.df_ratings['car_id'] = self.df_ratings['car_id'].astype(str)
        self.df_cars['id'] = self.df_cars['id'].astype(str)
        # Encode ID sang số nguyên (0, 1, 2...) để đưa vào Neural Net
        self.u_enc = LabelEncoder()
        self.i_enc = LabelEncoder()
        
        self.df_ratings['u_idx'] = self.u_enc.fit_transform(self.df_ratings['user_id'])
        # Fit trên toàn bộ xe để tránh lỗi xe mới
        self.i_enc.fit(self.df_cars['id'])
        try:
            self.df_ratings['i_idx'] = self.i_enc.transform(self.df_ratings['car_id'])
        except ValueError:
            # Fallback phòng trường hợp rating chứa xe đã bị xóa khỏi file csv gốc
            # Lọc bỏ các rating của xe không tồn tại
            valid_cars = set(self.df_cars['id'])
            self.df_ratings = self.df_ratings[self.df_ratings['car_id'].isin(valid_cars)]
            self.df_ratings['i_idx'] = self.i_enc.transform(self.df_ratings['car_id'])
        # 2. Encoders cho Feature Mới (Brand, Type, Persona)
        self.p_enc = LabelEncoder() # Persona
        self.b_enc = LabelEncoder() # Brand (Make)
        self.t_enc = LabelEncoder() # Car Type
        
        self.df_ratings['p_idx'] = self.p_enc.fit_transform(self.df_ratings['persona'])
        self.df_cars['b_idx'] = self.b_enc.fit_transform(self.df_cars['make'])
        self.df_cars['t_idx'] = self.t_enc.fit_transform(self.df_cars['car_type'])
        
        # 3. Chuẩn hóa dữ liệu số (MinMax Scaling thủ công để đưa về 0-1)
        # Price, Year, Power, Seats, TechScore
        cars = self.df_cars
        self.df_cars['norm_price'] = np.log1p(cars['price']) / np.log1p(cars['price'].max()) # Log để giảm chênh lệch giá
        self.df_cars['norm_year'] = (cars['year'] - 1990) / (2025 - 1990)
        self.df_cars['norm_power'] = cars['power'] / cars['power'].max()
        self.df_cars['norm_seats'] = cars['n_seats'] / 16.0
        self.df_cars['norm_tech'] = cars.get('tech_score', 0) / 10.0 # Nếu chưa có tech_score thì = 0

        # Lưu số lượng classes để init model
        self.dims = {
            'n_users': len(self.u_enc.classes_),
            'n_personas': len(self.p_enc.classes_),
            'n_items': len(self.i_enc.classes_),
            'n_brands': len(self.b_enc.classes_),
            'n_types': len(self.t_enc.classes_)
        }
        
        # Merge thông tin xe vào bảng ratings để lúc train có dữ liệu
        # Chỉ lấy các cột cần thiết
        car_features = self.df_cars[['id', 'b_idx', 't_idx', 'norm_price', 'norm_year', 'norm_power', 'norm_seats', 'norm_tech']]
        self.train_df = pd.merge(self.df_ratings, car_features, left_on='car_id', right_on='id', how='left')

    def load_feedback_data(self):
            """
            [BỔ SUNG] Đọc dữ liệu feedback thực tế từ user_interactions_log.csv
            để gộp vào dữ liệu training gốc.
            """
            log_path = "user_interactions_log.csv"
            if not os.path.exists(log_path):
                return

            try:
                # Đọc log
                df_log = pd.read_csv(log_path)
                
                # Chỉ lấy các hành động có điểm số (like, contact, view...)
                df_feedback = df_log[['user_id', 'car_id', 'implied_rating']].rename(columns={'implied_rating': 'rating'})
                
                # Gán persona mặc định (vì log chưa có persona, ta coi là 'Mixed')
                df_feedback['persona'] = 'Mixed' 
                
                # Ép kiểu dữ liệu cho khớp với bảng train gốc
                df_feedback['user_id'] = df_feedback['user_id'].astype(str) # ID user thật thường là string (uuid)
                df_feedback['car_id'] = df_feedback['car_id'].astype(str)
                
                print(f"4. [Feedback] Đã nạp thêm {len(df_feedback)} tương tác thực tế vào bộ nhớ.")
                
                # Gộp vào df_ratings hiện tại (chỉ trong RAM, chưa lưu đè file gốc để tránh lỗi)
                self.df_ratings = pd.concat([self.df_ratings, df_feedback], ignore_index=True)
                
                
            except Exception as e:
                print(f"⚠️ Lỗi đọc feedback log: {e}")

    def _load_or_train_models(self):
        path_svd = f"{self.cp_dir}/svd.pkl"
        path_torch = f"{self.cp_dir}/twotower.pth"

        # Check xem đã train chưa
        if os.path.exists(path_svd) and os.path.exists(path_torch):
            print("3. [Model] ✅ Load model đã train từ checkpoint.")
            _, self.svd = dump.load(path_svd)
            self.torch_model = AdvancedTwoTowerNet(
                self.dims['n_users'], 
                self.dims['n_personas'],
                self.dims['n_items'], 
                self.dims['n_brands'], 
                self.dims['n_types']
            ).to(device)
            self.torch_model.load_state_dict(torch.load(path_torch, map_location=device))
            self.torch_model.eval()
            
            # Đánh giá lại nhanh
            self.evaluate_model()
        else:
            print("3. [Model] ⚠️ Chưa có model. Bắt đầu Train mới...")
            self._train_models(path_svd, path_torch)

    def _train_models(self, path_svd, path_torch):
        # A. Train SVD (Matrix Factorization)
        print("   -> Training SVD...")
        reader = Reader(rating_scale=(1, 5))
        data = SurpriseDataset.load_from_df(self.df_ratings[['user_id', 'car_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.svd.fit(trainset)
        dump.dump(path_svd, algo=self.svd)

        # --- TRAIN TWO TOWER ---
        print("   -> Training Advanced Two-Tower Neural Network...")
        
        # Khởi tạo model với đầy đủ tham số kích thước
        self.torch_model = AdvancedTwoTowerNet(
            self.dims['n_users'], self.dims['n_personas'],
            self.dims['n_items'], self.dims['n_brands'], self.dims['n_types']
        ).to(device)
        self.torch_model.train()
        
        # Chuẩn bị Tensor Input
        # User Input: [u_idx, p_idx]
        user_feats = self.train_df[['u_idx', 'p_idx']].values
        
        self.train_df['i_idx_mapped'] = self.i_enc.transform(self.train_df['car_id'])
        
        item_cols = ['i_idx_mapped', 'b_idx', 't_idx', 'norm_price', 'norm_year', 'norm_power', 'norm_seats', 'norm_tech']
        item_feats = self.train_df[item_cols].values
        
        targets = self.train_df['rating'].values

        # Tạo Tensor
        u_tensor = torch.tensor(user_feats, dtype=torch.float32) # Sẽ convert long bên trong model
        i_tensor = torch.tensor(item_feats, dtype=torch.float32)
        r_tensor = torch.tensor(targets, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(u_tensor, i_tensor, r_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.torch_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(20): # Train 20 epochs
            total_loss = 0
            for u_batch, i_batch, r_batch in loader:
                u_batch, i_batch, r_batch = u_batch.to(device), i_batch.to(device), r_batch.to(device)
                optimizer.zero_grad()
                preds = self.torch_model(u_batch, i_batch)
                loss = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"      Epoch {epoch+1}/20 - Loss: {total_loss/len(loader):.4f}")

        torch.save(self.torch_model.state_dict(), path_torch)
        print("   -> Train xong. Đã lưu model.")
        
        # C. Đánh giá ngay sau khi train
        self.evaluate_model()
    def retrain_model(self):
        """
        [BỔ SUNG] Hàm kích hoạt huấn luyện lại model ngay lập tức.
        Quy trình:
        1. Đọc lại file log feedback.
        2. Gộp vào dữ liệu cũ.
        3. Train lại SVD và Neural Network.
        4. Cập nhật model nóng trong RAM.
        """
        print("🔄 [System] Bắt đầu quy trình Retrain...")
        
        # 1. Nạp dữ liệu mới nhất
        self.load_feedback_data()
        
        # 2. Định nghĩa lại đường dẫn lưu checkpoint
        path_svd = f"{self.cp_dir}/svd.pkl"
        path_torch = f"{self.cp_dir}/twotower.pth"
        
        # 3. Gọi hàm train (Hàm này sẽ update self.svd và self.torch_model)
        self._train_models(path_svd, path_torch)
        
        # 4. Re-build similarity matrix (để cập nhật tính năng xe tương tự)
        self._build_item_similarity()
        
        print("✅ [System] Retrain hoàn tất! Model đã cập nhật kiến thức mới.")
        return True
    def _build_item_similarity(self):
        """
        Xây dựng ma trận tương đồng giữa các xe (Item-Item CF)
        Kiến thức từ Slide: Dùng Cosine Similarity trên ma trận User-Item
        """
        print("4. [Sim] Đang tính toán ma trận Item-Item Similarity (Slide Knowledge)...")
        # Tạo ma trận thưa: Hàng = Xe, Cột = User, Giá trị = Rating
        pivot = self.df_ratings.pivot_table(index='car_id', columns='user_id', values='rating').fillna(0)
        
        # Tính Cosine Similarity giữa các XE
        self.item_sim_matrix = cosine_similarity(pivot)
        
        # Lưu index để tra cứu ngược
        self.sim_car_ids = pivot.index.tolist()
        print("   -> Đã xây dựng xong ma trận tương đồng.")

    def evaluate_model(self, k=5):
        """
        Đánh giá toàn diện mô hình:
        1. Regression Metrics: RMSE, MAE (Dự đoán điểm có chuẩn không?)
        2. Ranking Metrics: Precision@K, Recall@K (Top K xe gợi ý có 'chất' không?)
        3. Segment Analysis: Đánh giá riêng từng nhóm Persona.
        """
        print("\n" + "="*60)
        print("📊 BÁO CÁO ĐÁNH GIÁ HIỆU QUẢ MÔ HÌNH (ADVANCED METRICS)")
        print("="*60)
        
        # Đảm bảo cột i_idx tồn tại (phòng hờ)
        if 'i_idx' not in self.df_ratings.columns:
             self.df_ratings['i_idx'] = self.i_enc.transform(self.df_ratings['car_id'])

        # 1. Chuẩn bị dữ liệu Test (20%)
        # Lấy mẫu ngẫu nhiên
        test_set = self.df_ratings.sample(frac=0.2, random_state=42)

        # Merge thêm thông tin xe (Brand, Type, Specs...) vào test_set
        # Để có đủ dữ liệu đầu vào cho Item Tower
        cols_to_merge = ['id', 'b_idx', 't_idx', 'norm_price', 'norm_year', 'norm_power', 'norm_seats', 'norm_tech']
        test_set = pd.merge(test_set, self.df_cars[cols_to_merge], left_on='car_id', right_on='id', how='left')

        y_true = []
        y_pred = []
        
        # Gom nhóm theo User để tính Ranking Metrics
        user_est_true = defaultdict(list)
        
        # Gom nhóm theo Persona để đánh giá phân khúc
        persona_metrics = defaultdict(lambda: {'true': [], 'pred': []})

        print(f"   -> Đang kiểm tra trên {len(test_set)} mẫu test...")
        
        self.torch_model.eval()
        
        with torch.no_grad():
            for _, row in test_set.iterrows():
                try:
                    uid = int(row['u_idx'])
                    cid = int(row['i_idx'])
                    persona = row['persona']
                    real_rating = row['rating']
                    
                    # --- 1. SVD PREDICTION ---
                    svd_pred = self.svd.predict(row['user_id'], row['car_id']).est
                    
                    # --- 2. TWO-TOWER PREDICTION ---
                    # Chuẩn bị User Input [u_idx, p_idx] -> Shape [1, 2]
                    u_vals = [uid, row['p_idx']]
                    u_t = torch.tensor([u_vals], dtype=torch.float32).to(device) 
                    
                    # Chuẩn bị Item Input [i_idx, b_idx, t_idx, price, year, power, seats, tech] -> Shape [1, 8]
                    # Phải lấy đúng thứ tự như lúc train
                    i_vals = [
                        cid, 
                        row['b_idx'], 
                        row['t_idx'], 
                        row['norm_price'], 
                        row['norm_year'], 
                        row['norm_power'], 
                        row['norm_seats'], 
                        row['norm_tech']
                    ]
                    i_t = torch.tensor([i_vals], dtype=torch.float32).to(device)
                    
                    # Dự đoán
                    dl_pred = float(self.torch_model(u_t, i_t).cpu().item())
                    
                    # --- 3. COMBINE ---
                    final_pred = 0.4 * svd_pred + 0.6 * dl_pred
                    
                    # Lưu kết quả
                    y_true.append(real_rating)
                    y_pred.append(final_pred)
                    
                    # Lưu cho Ranking
                    user_est_true[row['user_id']].append((final_pred, real_rating))
                    
                    # Lưu cho Persona Analysis
                    persona_metrics[persona]['true'].append(real_rating)
                    persona_metrics[persona]['pred'].append(final_pred)
                    
                except Exception as e:
                    # Bỏ qua nếu có lỗi dữ liệu nhỏ lẻ
                    continue

        # ======================================================================
        # A. ĐÁNH GIÁ ĐỘ CHÍNH XÁC (REGRESSION)
        # ======================================================================
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print(f"\n1️⃣  ĐỘ CHÍNH XÁC ĐIỂM SỐ (ACCURACY):")
        print(f"   - RMSE (Sai số bình phương trung bình): {rmse:.4f} ⭐ (Thấp hơn 1.0 là Tốt)")
        print(f"   - MAE  (Sai số tuyệt đối trung bình)  : {mae:.4f} ⭐ (Lệch trung bình bao nhiêu điểm)")

        # ======================================================================
        # B. ĐÁNH GIÁ XẾP HẠNG (RANKING - PRECISION@K)
        # ======================================================================
        # Định nghĩa: "Relevant" (Thích) là rating >= 4.0
        precisions = []
        recalls = []
        
        for uid, user_ratings in user_est_true.items():
            # Sắp xếp các xe đã test theo điểm dự đoán giảm dần
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            
            # Lấy Top K
            top_k_items = user_ratings[:k]
            
            # Đếm số lượng xe thực sự thích trong Top K (True Rating >= 4.0)
            n_rel_and_rec = sum(1 for (_, true_r) in top_k_items if true_r >= 4.0)
            
            # Tổng số xe thực sự thích trong toàn bộ test set của user này
            n_rel = sum(1 for (_, true_r) in user_ratings if true_r >= 4.0)
            
            # Precision@K
            precisions.append(n_rel_and_rec / k if k > 0 else 0)
            
            # Recall@K
            recalls.append(n_rel_and_rec / n_rel if n_rel > 0 else 0)
            
        p_at_k = sum(precisions) / len(precisions) if precisions else 0
        r_at_k = sum(recalls) / len(recalls) if recalls else 0
        
        print(f"\n2️⃣  CHẤT LƯỢNG GỢI Ý (RANKING @{k}):")
        print(f"   - Precision@{k}: {p_at_k:.2%} (Tỉ lệ xe user thích trong top {k})")
        print(f"   - Recall@{k}   : {r_at_k:.2%} (Tỉ lệ tìm thấy xe ngon trong kho)")

        # ======================================================================
        # C. PHÂN TÍCH THEO NHÓM NGƯỜI DÙNG (SEGMENTATION)
        # ======================================================================
        print(f"\n3️⃣  HIỆU NĂNG THEO PERSONA (SEGMENTATION):")
        print(f"   {'Persona':<15} | {'RMSE':<10} | {'MAE':<10} | {'Trạng thái'}")
        print("-" * 55)
        
        for p, data in persona_metrics.items():
            if not data['true']: continue
            p_rmse = sqrt(mean_squared_error(data['true'], data['pred']))
            p_mae = mean_absolute_error(data['true'], data['pred'])
            
            status = "✅ Tốt" if p_rmse < 1.0 else ("⚠️ Khá" if p_rmse < 1.2 else "❌ Kém")
            print(f"   {p:<15} | {p_rmse:.4f}     | {p_mae:.4f}     | {status}")
            
        print("="*60 + "\n")
        return rmse

    def recommend(self, profile_dict, top_k=5):
        """
        Hàm gợi ý tối ưu hóa cho dữ liệu thực tế (Scraped CSV).
        Quy trình:
        1. Lọc cứng (Specific Brands -> Text Search -> Technical Specs -> Persona).
        2. Cơ chế Fallback thông minh (Nếu lọc hết xe -> Nới lỏng điều kiện).
        3. Chấm điểm Hybrid (Vector Cosine + SVD Rating + Rule-based Boost).
        """
        
        # --- BƯỚC 1: KHỞI TẠO BỘ LỌC ---
        # Ta tạo một bản sao để lọc dần
        filtered_df = self.df_cars.copy()
        
        # Flag để biết xem có đang lọc quá gắt không
        initial_count = len(filtered_df)
        
        # ----------------------------------------------------------------------
        # A. LỌC THEO HÃNG CỤ THỂ (Ưu tiên cao nhất - Dành cho lệnh So sánh)
        # ----------------------------------------------------------------------
        if 'specific_brands' in profile_dict and profile_dict['specific_brands']:
            target_brands = [b.lower() for b in profile_dict['specific_brands']]
            # Tìm tương đối: Ví dụ user nói "Merc" thì vẫn ra "Mercedes-Benz"
            filtered_df = filtered_df[filtered_df['make'].str.lower().apply(
                lambda x: any(t in x for t in target_brands)
            )]

        # ----------------------------------------------------------------------
        # B. TÌM KIẾM TỪ KHÓA (SEMANTIC SEARCH THÔ)
        # ----------------------------------------------------------------------
        if 'search_query' in profile_dict:
            q = profile_dict['search_query'].lower()
            
            # 1. Tìm theo loại xe
            if 'suv' in q or 'gầm cao' in q or '7 chỗ' in q:
                # Tìm trong tên xe hoặc số chỗ
                filtered_df = filtered_df[
                    (filtered_df['name'].str.lower().str.contains('suv|cross|fortuner|everest|santa|sorento')) | 
                    (filtered_df['n_seats'] >= 7)
                ]
            elif 'sedan' in q:
                filtered_df = filtered_df[filtered_df['n_seats'] <= 5]
            elif 'bán tải' in q or 'pickup' in q:
                filtered_df = filtered_df[filtered_df['name'].str.lower().str.contains('ranger|triton|hilux|navara')]

            # 2. Tìm theo tính năng (trong description hoặc features)
            # Ví dụ: "xe có cửa sổ trời"
            keywords = ['cửa sổ trời', 'camera 360', 'ghế da', 'turbo']
            for kw in keywords:
                if kw in q:
                    filtered_df = filtered_df[
                        filtered_df['features'].str.lower().str.contains(kw, na=False) |
                        filtered_df['description'].str.lower().str.contains(kw, na=False)
                    ]

        # ----------------------------------------------------------------------
        # C. LỌC KỸ THUẬT (TECHNICAL SPECS)
        # ----------------------------------------------------------------------
        # [BỔ SUNG] Lọc theo giá trần từ câu chat (nếu có)
        if 'max_price_override' in profile_dict and profile_dict['max_price_override'] > 0:
            max_p = profile_dict['max_price_override']
            # Cho phép dung sai 10% (xe đắt hơn xíu vẫn lấy)
            filtered_df = filtered_df[filtered_df['price'] <= max_p * 1.1]
        # Năm sản xuất
        if 'min_year' in profile_dict:
            filtered_df = filtered_df[filtered_df['year'] >= profile_dict['min_year']]
        
        # Sức mạnh động cơ (Mã lực)
        if 'min_power' in profile_dict:
            filtered_df = filtered_df[filtered_df['power'] >= profile_dict['min_power']]
            
        # Nhiên liệu (Map từ input User sang dữ liệu CSV 'Xăng'/'Dầu'/'Điện')
        if 'fuel_type' in profile_dict:
            req = profile_dict['fuel_type'].lower()
            if req in ['xăng', 'petrol']:
                filtered_df = filtered_df[filtered_df['fuelType'].str.lower() == 'xăng']
            elif req in ['dầu', 'diesel']:
                filtered_df = filtered_df[filtered_df['fuelType'].str.lower() == 'dầu']
            elif req in ['điện', 'electric', 'ev']:
                filtered_df = filtered_df[filtered_df['fuelType'].str.lower().isin(['điện', 'hybrid'])]

        # ----------------------------------------------------------------------
        # D. LỌC THEO PERSONA (Nếu chưa bị lọc bởi Brand/Query)
        # ----------------------------------------------------------------------
        # Chỉ áp dụng nếu danh sách còn nhiều xe (> 10) để tránh filtered_df bị rỗng
        if len(filtered_df) > 10:
            persona = profile_dict.get('persona', 'Family')
            
            if persona == 'Student': # Ưu tiên xe rẻ
                filtered_df = filtered_df[filtered_df['price_code'] <= 2] 
            
            elif persona == 'Family': # Ưu tiên xe rộng, đời không quá sâu
                filtered_df = filtered_df[filtered_df['is_family'] == 1]
                filtered_df = filtered_df[filtered_df['year'] >= 2015]
            
            elif persona == 'Boss': # Ưu tiên xe sang
                filtered_df = filtered_df[filtered_df['is_luxury'] == 1]
            
            elif persona == 'Racer': # Ưu tiên xe mạnh
                filtered_df = filtered_df[filtered_df['power'] > 150]
        print(f"🔍 [Filter Stats] Ban đầu: {initial_count} xe -> Sau khi lọc: {len(filtered_df)} xe")
        
        if len(filtered_df) == 0:
            print("   ⚠️ Cảnh báo: Bộ lọc quá chặt, không còn xe nào khớp!")
        elif len(filtered_df) < initial_count * 0.1:
            print("   ⚠️ Cảnh báo: Đã lọc bỏ hơn 90% dữ liệu, kết quả có thể bị hạn chế.")

        # ----------------------------------------------------------------------
        # E. FEATURE MATCHING (Lọc mềm bằng từ khóa)
        # ----------------------------------------------------------------------
        # Nếu user yêu cầu tính năng cụ thể (VD: Cửa sổ trời), ta ưu tiên lọc.
        # Nhưng nếu lọc xong còn quá ít xe (<3), ta sẽ bỏ qua bước này (Fallback).
        
        req_features = profile_dict.get('features', [])
        if req_features:
            temp_df = filtered_df.copy()
            # Map từ khóa AI trả về sang cột dữ liệu (đã tạo ở bước clean data)
            feature_map = {
                'sunroof': 'has_sunroof',
                'adas': 'has_adas',
                '360_camera': 'has_360',
                'leather': 'has_leather',
                'smartkey': 'has_smartkey'
            }
            
            for req in req_features:
                col = feature_map.get(req)
                if col and col in temp_df.columns:
                    # Lọc xe có tính năng này
                    temp_df = temp_df[temp_df[col] == 1]
            
            # Chỉ áp dụng nếu còn xe (tránh trả về rỗng)
            if len(temp_df) >= 3:
                filtered_df = temp_df
                print(f"✨ [Engine] Đã lọc theo tính năng: {req_features}")
            else:
                print(f"⚠️ [Engine] Không tìm thấy xe có {req_features}, bỏ qua filter này.")
        # ----------------------------------------------------------------------
        # E. CƠ CHẾ FALLBACK (CỨU CÁNH)
        # ----------------------------------------------------------------------
        # Nếu lọc xong mà còn ít hơn 3 xe -> Nới lỏng điều kiện
        if len(filtered_df) < 3:
            # Reset lại tập lọc (Lấy lại toàn bộ xe khớp Brand/Query nhưng bỏ qua Năm/Specs)
            # Hoặc tệ nhất là lấy toàn bộ DB
            candidates_ids = self.df_cars['id'].tolist()
        else:
            candidates_ids = filtered_df['id'].tolist()

        # Sampling: Nếu còn quá nhiều xe (>200), lấy ngẫu nhiên 200 để tính toán cho nhanh
        if len(candidates_ids) > 200:
            candidates_ids = np.random.choice(candidates_ids, 200, replace=False)

        # ----------------------------------------------------------------------
        # F. SCORING ENGINE (CHẤM ĐIỂM)
        # ----------------------------------------------------------------------
        
        # 1. Xác định Proxy User
        persona = profile_dict.get('persona', 'Family')
        proxy_users = self.df_ratings[self.df_ratings['persona'] == persona]['user_id'].unique()
        user_id = proxy_users[0] if len(proxy_users) > 0 else self.df_ratings['user_id'].iloc[0]
        
        # 2. Chuẩn bị Tensor
        u_idx = self.u_enc.transform([user_id])[0]
        try:
            c_idxs = self.i_enc.transform(candidates_ids)
        except:
            # Fallback nếu gặp ID lạ (xe mới cào thêm mà chưa train)
            return self.df_cars[self.df_cars['id'].isin(candidates_ids)].head(top_k)

        # Chuẩn bị User Features [u_idx, p_idx]
        p_idx = self.p_enc.transform([persona])[0]
        u_input = torch.tensor([[u_idx, p_idx]] * len(candidates_ids)).to(device) # Shape [N, 2]
        
        # Chuẩn bị Item Features
        # Lấy thông tin các xe candidates từ df_cars
        candidate_cars = self.df_cars[self.df_cars['id'].isin(candidates_ids)].set_index('id')
        candidate_cars = candidate_cars.reindex(candidates_ids) # Đảm bảo đúng thứ tự
        
        # Lấy các cột features
        c_idxs = self.i_enc.transform(candidates_ids)
        b_idxs = candidate_cars['b_idx'].values
        t_idxs = candidate_cars['t_idx'].values
        numerics = candidate_cars[['norm_price', 'norm_year', 'norm_power', 'norm_seats', 'norm_tech']].values
        
        # Gộp lại thành tensor [N, 8]
        # [i_idx, b_idx, t_idx, price, year, power, seats, tech]
        i_data = np.column_stack([c_idxs, b_idxs, t_idxs, numerics])
        i_input = torch.tensor(i_data, dtype=torch.float32).to(device)
        
        self.torch_model.eval()
        with torch.no_grad():
            
            # Forward user part
            u_vec = self.torch_model.user_emb(u_input[:, 0].long())
            p_vec = self.torch_model.persona_emb(u_input[:, 1].long())
            user_rep = self.torch_model.user_layers(torch.cat([u_vec, p_vec], dim=1))
            
            # Forward item part
            i_vec = self.torch_model.item_emb(i_input[:, 0].long())
            b_vec = self.torch_model.brand_emb(i_input[:, 1].long())
            t_vec = self.torch_model.type_emb(i_input[:, 2].long())
            n_vec = F.relu(self.torch_model.numeric_trans(i_input[:, 3:].float()))
            item_rep = self.torch_model.item_layers(torch.cat([i_vec, b_vec, t_vec, n_vec], dim=1))
            
            # Tính Cosine & Rating
            cosine_scores = F.cosine_similarity(user_rep, item_rep).cpu().numpy()
            
            # Tính Dot Product -> Predicted Rating
            dl_ratings = (user_rep * item_rep).sum(dim=1).cpu().numpy()

        # Lấy danh sách xe user đã Like trong phiên này - feedback
        liked_car_ids = profile_dict.get('liked_history', [])

        # 4. Tổng hợp kết quả
        results = []
        for idx, car_id in enumerate(candidates_ids):
            # Lấy thông tin xe để Boost điểm
            car_info = self.df_cars[self.df_cars['id'] == car_id].iloc[0]
            
            # Điểm cơ bản
            svd_rating = self.svd.predict(user_id, car_id).est
            dl_rating = float(dl_ratings[idx])
            final_rating = 0.4 * svd_rating + 0.6 * dl_rating
            
            # Tính Match Score (%)
            raw_match = (cosine_scores[idx] + 1) / 2 # Normalize 0-1
            match_percent = raw_match * 100
            # --- [NÂNG CẤP] RULE-BASED BOOSTING ---
            
            # 1. Boost theo Hiệu suất (Nếu user thích xe mạnh)
            if profile_dict.get('high_performance', False):
                # Mã lực > 180 là mạnh
                if car_info['power'] > 180:
                    match_percent += 15
                    final_rating += 0.8
                # Turbo thường mạnh
                if 'turbo' in str(car_info['engine']).lower():
                    match_percent += 5

            # 2. Boost theo Tình trạng (Xe lướt)
            if profile_dict.get('car_condition') == 'like_new':
                # Xe dưới 3 tuổi và ODO thấp
                if car_info['age'] <= 3 and car_info['mileage'] < 40000:
                    match_percent += 10
                    final_rating += 0.5
            # --- RULE-BASED BOOSTING (Cộng điểm thưởng) ---
            # Thưởng cho xe đời mới (> 2022)
            if car_info['year'] >= 2022: 
                match_percent += 5
                final_rating += 0.2
            
            # Thưởng nếu đúng Brand yêu thích (nếu có trong profile)
            if 'preferredBrands' in profile_dict and profile_dict['preferredBrands']:
                fav_brands = [b.lower() for b in profile_dict['preferredBrands']]
                if str(car_info['make']).lower() in fav_brands:
                    match_percent += 10
                    final_rating += 0.5
            # --- REAL-TIME FEEDBACK BOOSTING ---
            # Nếu xe này tương đồng với xe user vừa Like -> Cộng điểm cực mạnh
            if liked_car_ids:
                # Kiểm tra xem xe hiện tại (car_id) có giống xe đã like không
                # Dùng ma trận item_sim_matrix đã tính
                for liked_id in liked_car_ids:
                    if liked_id in self.sim_car_ids and car_id in self.sim_car_ids:
                        # Lấy index
                        idx_curr = self.sim_car_ids.index(car_id)
                        idx_liked = self.sim_car_ids.index(liked_id)
                        
                        # Lấy độ tương đồng (0 -> 1)
                        sim_score = self.item_sim_matrix[idx_curr][idx_liked]
                        
                        if sim_score > 0.6: # Nếu giống > 60%
                            boost = sim_score * 15 # Cộng tối đa 15% match
                            match_percent += boost
                            final_rating += (sim_score * 1.0) # Cộng tối đa 1 điểm rating
                            # Break để không cộng dồn quá nhiều nếu like nhiều xe giống nhau
                            break 
            # ---------------------------------------------
            # Clip kết quả
            match_percent = min(99, int(match_percent))
            
            results.append({
                'id': car_id, 
                'score': final_rating, 
                'match_percent': match_percent
            })
            
        # Sắp xếp theo điểm cao nhất
        res_df = pd.DataFrame(results).sort_values('score', ascending=False).head(top_k)
        
        # Merge lại để lấy full thông tin xe
        final_df = pd.merge(res_df, self.df_cars, on='id')
        return final_df
    def _get_content_based_similar_cars(self, car_id, top_k=5):
        """
        FALLBACK: Tìm xe tương tự dựa trên thông số kỹ thuật (dùng khi chưa có rating).
        Logic: Cùng phân khúc (Body Type) -> Cùng tầm giá -> Cùng hãng (ưu tiên).
        """
        # 1. Lấy thông tin xe gốc
        try:
            # Đảm bảo ID là string để so sánh
            car_id = str(car_id)
            target_car = self.df_cars[self.df_cars['id'] == car_id].iloc[0]
        except IndexError:
            return pd.DataFrame() # Xe không tồn tại trong kho

        # 2. Lọc xe cùng kiểu dáng (Body Type)
        # Giả sử đã có cột 'car_type' từ hàm process(), nếu chưa thì dùng logic đơn giản
        target_type = target_car.get('car_type', '')
        
        # Lấy danh sách ứng viên (trừ chính nó)
        candidates = self.df_cars[self.df_cars['id'] != car_id].copy()
        
        # Tính điểm tương đồng (Distance Metric)
        # Công thức: 
        # - Cùng Body Type: +40đ
        # - Cùng Hãng: +20đ
        # - Chênh lệch giá: Tối đa 40đ (càng gần càng cao)
        
        def calculate_similarity(row):
            score = 0
            
            # 1. Body Type (Quan trọng nhất)
            if row.get('car_type') == target_type:
                score += 40
            
            # 2. Brand
            if row['make'] == target_car['make']:
                score += 20
                
            # 3. Price Similarity (Max 40 điểm)
            # Tính % chênh lệch giá. Ví dụ lệch 0% -> 40đ, lệch 50% -> 0đ
            try:
                price_diff = abs(row['price'] - target_car['price'])
                percent_diff = price_diff / (target_car['price'] + 1) # +1 tránh chia 0
                price_score = max(0, 40 * (1 - percent_diff * 2)) # Lệch 50% là hết điểm
                score += price_score
            except:
                pass
                
            # 4. Year Similarity (Bonus nhẹ)
            year_diff = abs(row['year'] - target_car['year'])
            if year_diff <= 2: score += 5
            
            return score

        candidates['sim_score'] = candidates.apply(calculate_similarity, axis=1)
        
        # Lấy top K xe có điểm cao nhất
        top_candidates = candidates.sort_values('sim_score', ascending=False).head(top_k)
        
        print(f"   ✨ [Content-Based] Tìm thấy {len(top_candidates)} xe tương tự theo thông số.")
        return top_candidates

    def get_similar_cars_item_based(self, car_id, top_k=3):
        """
        HYBRID SIMILARITY:
        1. Thử tìm bằng Item-Item CF (Hành vi người dùng - Chính xác nhất).
        2. Nếu không có (xe mới), Fallback sang Content-Based (Thông số kỹ thuật).
        """
        car_id = str(car_id)
        print(f"\n🔍 Tìm xe tương tự cho xe ID: {car_id}")
        
        cf_results = pd.DataFrame()
        
        # --- CÁCH 1: COLLABORATIVE FILTERING (Ưu tiên) ---
        if hasattr(self, 'sim_car_ids') and car_id in self.sim_car_ids:
            try:
                # Lấy index
                idx = self.sim_car_ids.index(car_id)
                # Lấy vector tương đồng
                sim_scores = self.item_sim_matrix[idx]
                # Sort lấy index cao nhất (trừ chính nó)
                top_indices = sim_scores.argsort()[-(top_k+1):-1][::-1]
                similar_ids = [self.sim_car_ids[i] for i in top_indices]
                
                cf_results = self.df_cars[self.df_cars['id'].isin(similar_ids)]
                print(f"   ✅ [CF] Tìm thấy {len(cf_results)} xe dựa trên hành vi người dùng.")
            except Exception as e:
                print(f"   ⚠️ Lỗi CF: {e}")

        # --- CÁCH 2: CONTENT-BASED (Fallback hoặc Bổ sung) ---
        # Nếu CF không trả về đủ số lượng xe (ví dụ top_k=3 mà CF chỉ ra 0 hoặc 1 xe)
        # Chúng ta sẽ tìm thêm bằng Content-Based để lấp đầy
        if len(cf_results) < top_k:
            needed = top_k - len(cf_results)
            print(f"   ⚠️ CF chưa đủ (có {len(cf_results)}/{top_k}), tìm thêm bằng Content-Based...")
            
            cb_results = self._get_content_based_similar_cars(car_id, top_k=needed + 5) # Lấy dư ra để lọc trùng
            
            # Loại bỏ xe đã có trong CF
            if not cf_results.empty:
                cb_results = cb_results[~cb_results['id'].isin(cf_results['id'])]
            
            # Gộp lại: CF lên đầu, Content-Based theo sau
            final_results = pd.concat([cf_results, cb_results.head(needed)])
            return final_results
            
        return cf_results

# --- MAIN TEST ---
if __name__ == "__main__":
    recsys = CarRecommendationSystem()
    
    # Test Evaluate
    recsys.evaluate_model()
    
    # Test Recommend cho Sinh viên
    print("\n--- TEST RECOMMEND: STUDENT ---")
    profile = {'persona': 'Student'}
    print(recsys.recommend(profile))
    
    # Test Item-Item (Slide)
    print("\n--- TEST SIMILAR CARS (ITEM-ITEM) ---")
    sample_car = recsys.df_cars['id'].iloc[0]
    print(f"Tìm xe giống xe ID {sample_car}:")
    print(recsys.get_similar_cars_item_based(sample_car))