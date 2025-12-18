# ==============================================================================
# FILE: api.py
# CHỨC NĂNG: Backend Server (FastAPI) + LLM Reranker + Feedback Loop
# ==============================================================================
import re
import json
import csv
import os
import uvicorn
import random
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi import BackgroundTasks
from difflib import SequenceMatcher
from config import GEMINI_API_KEY
# Import Core Engine & Gemini Client
try:
    from recommender_engine import CarRecommendationSystem
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    exit()

# ==============================================================================
# 1. CẤU HÌNH & KHỞI TẠO
# ==============================================================================


app = FastAPI(title="Smart Car RecSys API")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo Hệ thống Gợi ý & AI Client
print("⏳ Đang khởi động hệ thống...")
try:
    # Load Engine với file CSV mới
    recsys = CarRecommendationSystem(csv_path="scraped_cars.csv") 
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Hệ thống đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi khởi tạo: {e}")
    recsys = None
# Cấu hình file log feedback
FEEDBACK_FILE = "user_interactions_log.csv"

# Khởi tạo file CSV nếu chưa tồn tại
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header chuẩn để sau này train lại model
        writer.writerow(["timestamp", "user_id", "car_id", "action", "implied_rating"])

# Database giả lập lưu lịch sử tương tác (User-User CF foundation)
user_interactions = {} 
# ==============================================================================
# 2. DATA MODELS (Pydantic)
# ==============================================================================
def log_feedback_to_csv(user_id: str, car_id: str, action: str):
    """
    Hàm chạy ngầm: Ghi log tương tác vào CSV để sau này retrain model.
    Quy đổi:
    - like    -> Rating 5.0
    - dislike -> Rating 1.0
    - view    -> Rating 3.0 (Ví dụ xem chi tiết xe)
    """
    # 1. Quy đổi hành động sang điểm số (Implicit Feedback)
    rating_map = {
        "like": 5.0,
        "dislike": 1.0,
        "view": 3.0,
        "contact": 5.0 # Nếu user bấm nút liên hệ
    }
    
    implied_rating = rating_map.get(action, 0)
    
    # 2. Ghi vào file
    try:
        with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                user_id,
                car_id,
                action,
                implied_rating
            ])
        print(f"💾 [System] Đã lưu feedback: {user_id} -> {action} -> {car_id}")
    except Exception as e:
        print(f"❌ Lỗi ghi file feedback: {e}")

    # 3. Cập nhật bộ nhớ Session (để dùng cho tính năng 'User-User' tức thì)
    # Nếu user like, ta lưu lại để gợi ý xe tương tự ngay trong phiên đó
    if action == "like":
        if user_id not in user_interactions:
            user_interactions[user_id] = []
        if car_id not in user_interactions[user_id]:
            user_interactions[user_id].append(car_id)

class UserProfileReq(BaseModel):
    age: int = 25
    income: int = 10000000
    maritalStatus: str = "single"      # 'single' | 'married'
    purpose: str = "commute"           # 'commute' | 'travel' | 'service' | 'family'
    priceRange: Optional[List[int]] = None  # [min, max]
    preferredBrands: Optional[List[str]] = []
    transmission: Optional[str] = "any"     # 'any' | 'manual' | 'automatic'

class ChatRequest(BaseModel):
    message: str
    userProfile: UserProfileReq
    history: Optional[List[Dict[str, str]]] = []
    sessionId: Optional[str] = "guest"

class FeedbackRequest(BaseModel):
    user_id: str 
    car_id: str
    action: str # "like" | "dislike"
# ==============================================================================
# 3. HELPER FUNCTIONS: MAPPING & LLM LOGIC
# ==============================================================================

def map_car_to_frontend(row, match_score=0):
    """
    Chuyển đổi 1 dòng dữ liệu từ CSV (phẳng) sang JSON (lồng nhau) cho FE.
    Xử lý các trường thiếu hoặc định dạng lại.
    """
    # Xử lý các giá trị an toàn
    def safe_str(val): return str(val) if pd.notna(val) else ""
    def safe_int(val, default=0): 
        try: return int(float(val)) if pd.notna(val) else default
        except: return default

    # 1. Thông tin cơ bản
    car_id = safe_str(row.get('id'))
    name = safe_str(row.get('name'))
    brand = safe_str(row.get('brand'))
    
    # 2. Xử lý Features (Trong CSV là chuỗi "A, B, C" -> List ["A", "B", "C"])
    raw_features = safe_str(row.get('features'))
    features_list = [f.strip() for f in raw_features.split(',')] if raw_features else []
    
    # Giới hạn hiển thị tối đa 5 feature nổi bật để UI không bị vỡ
    if len(features_list) > 5:
        features_list = features_list[:5]

    return {
        "id": car_id,
        "name": name,
        "brand": brand,
        "year": safe_int(row.get('year'), 2020),
        "price": safe_int(row.get('price'), 0),
        "image": safe_str(row.get('image')), 
        "seats": safe_int(row.get('seats'), 5),
        "transmission": safe_str(row.get('transmission')),
        "fuelType": safe_str(row.get('fuelType')),
        
        "matchScore": int(match_score),
        "matchReason": "Phù hợp với nhu cầu và sở thích của bạn.",
        
        # 3. Gom nhóm Specs (Nested Object)
        "specs": {
            "engine": safe_str(row.get('engine', 'N/A')),
            "horsepower": safe_int(row.get('horsepower') if 'horsepower' in row else row.get('power'), 100),
            "torque": safe_str(row.get('torque', 'N/A')),
            "fuelConsumption": safe_str(row.get('fuelConsumption', 'N/A')),
            "dimensions": safe_str(row.get('dimensions', 'N/A')),
            "weight": safe_str(row.get('weight', 'N/A'))
        },
        
        "description": safe_str(row.get('description', 'Đang cập nhật thông tin...')),
        "features": features_list
    }


# ==============================================================================
# 5. LLM RERANKER (GEMINI)
# ==============================================================================


def llm_rerank_and_explain(user_msg, user_profile, car_list):
    """
    💎 AI CONSULTANT (LỜI GIẢI THÍCH THÔNG MINH)
    Chức năng:
    1. Nhận vào Top 3 xe tốt nhất từ Engine (đã được sắp xếp theo điểm).
    2. Phân tích tâm lý người dùng (Psychological Profiling).
    3. Viết lời tư vấn bán hàng thuyết phục (Persuasive Copywriting) cho 3 xe này.
    """
    
    # 1. CHUẨN BỊ DỮ LIỆU (Chỉ mô tả 3 xe được truyền vào)
    cars_context = ""
    for i, car in enumerate(car_list):
        # Lấy tối đa 5 tính năng
        feats = ", ".join(car.get('features', [])[:5]) if car.get('features') else "Cơ bản"
        specs = car.get('specs', {})
        
        cars_context += (
            f"--- ỨNG VIÊN SỐ {i+1}: {car['name']} ---\n"
            f"- Thông số: {car['year']}, {car['brand']}, {car['seats']} chỗ, {car['transmission']}\n"
            f"- Giá: {car['price']:,} VNĐ\n"
            f"- Điểm phù hợp hệ thống chấm: {car.get('matchScore', 0)}/100\n"
            f"- Lý do kỹ thuật: {car.get('matchReason', '')}\n"
            f"- Tính năng: {feats}\n\n"
        )

    # 2. XÂY DỰNG PROMPT (Giữ nguyên phần Persona xịn xò)
    tone_instruction = "Chuyên nghiệp, tin cậy và khách quan."
    if user_profile.income > 25000000 or user_profile.age > 45:
        tone_instruction = "Sang trọng, lịch thiệp, tôn trọng đẳng cấp khách hàng (gọi là 'quý khách')."
    elif user_profile.age < 30:
        tone_instruction = "Trẻ trung, năng động, tập trung vào công nghệ, tốc độ và sự sành điệu."
    elif user_profile.purpose == "family":
        tone_instruction = "Ấm áp, quan tâm, nhấn mạnh sự an toàn, rộng rãi và tiện nghi cho gia đình."

    prompt = f"""
    [VAI TRÒ]
    Bạn là một chuyên gia tư vấn xe hơi cao cấp (AI Concierge) với 20 năm kinh nghiệm.
    
    [NHIỆM VỤ]
    Hệ thống tính toán kỹ thuật đã lọc ra 3 chiếc xe phù hợp nhất bên dưới.
    Nhiệm vụ của bạn KHÔNG PHẢI LÀ CHỌN LẠI, mà là viết một đoạn lời thoại tư vấn thật hay để giới thiệu 3 chiếc xe này tới khách hàng.

    [HỒ SƠ KHÁCH HÀNG]
    - Tuổi: {user_profile.age} | Thu nhập: {user_profile.income} USD/năm
    - Tình trạng hôn nhân: {user_profile.maritalStatus} | Mục đích: {user_profile.purpose}
    - Câu hỏi/Nhu cầu: "{user_msg}"

    [DANH SÁCH 3 XE TỐT NHẤT]
    {cars_context}

    [YÊU CẦU NỘI DUNG]
    1. Mở đầu: Chào hỏi theo giọng điệu {tone_instruction}.
    2. Phân tích nhanh: Nhắc khéo tại sao các xe này lại hợp với nhu cầu (ví dụ: "Vì anh cần xe gia đình an toàn nên tôi chọn...").
    3. Điểm nhấn: Nêu bật 1 ưu điểm "đắt giá" nhất của xe đứng đầu (Ứng viên số 1).
    4. Kết thúc: Mời khách xem chi tiết bên dưới.
    5. Độ dài: Ngắn gọn, súc tích (dưới 80 từ).

    [YÊU CẦU ĐẦU RA]
    Trả về định dạng JSON chuẩn (RFC 8259), KHÔNG Markdown:
    {{
        "analysis": "Lời tư vấn của bạn ở đây..."
    }}
    """

    # 3. GỌI API GEMINI
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", # Hoặc gemma-3-4b-it tùy bạn
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7, # Tăng nhiệt độ chút để văn phong tự nhiên hơn
                top_p=0.9,
            )
        )
        
        raw_text = response.text.strip()
        
        # 4. PARSING JSON
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        else:
            # Fallback nếu AI không trả JSON
            return {"analysis": raw_text}

    except Exception as e:
        print(f"⚠️ [AI Explain Error] {e}")
        return {"analysis": "Dưới đây là những lựa chọn tốt nhất được hệ thống tổng hợp dựa trên nhu cầu của bạn."}
    
def analyze_user_intent(message: str):
    """
    🧠 ADVANCED INTENT RECOGNITION SYSTEM (NLU Engine)
    Chức năng:
    1. Phân loại ý định chính xác (Search, Compare, Consult, Chitchat).
    2. Trích xuất Entities cực mạnh: Giá tiền (VNĐ), Dáng xe, Hộp số, Mục đích sử dụng.
    3. Chuẩn hóa dữ liệu đầu vào cho bộ lọc.
    """
    
    # Prompt kỹ thuật "Few-Shot Learning" để dạy AI cách xử lý các case khó
    prompt = f"""
    Bạn là một NLU Engine (Bộ hiểu ngôn ngữ tự nhiên) chuyên biệt cho ngành ô tô tại Việt Nam.
    Nhiệm vụ: Phân tích câu chat của khách hàng và trích xuất dữ liệu có cấu trúc (JSON).

    Câu chat: "{message}"
    [QUY TẮC ƯU TIÊN QUAN TRỌNG]
    - Nếu người dùng đưa ra yêu cầu cụ thể trong câu chat (ví dụ: "tìm xe Honda"), đây là **HARD CONSTRAINT**.
    - Các thông tin cũ (như user thích Toyota trong quá khứ) phải bị ghi đè bởi yêu cầu hiện tại.
    [QUY TẮC TRÍCH XUẤT]
    1. **Intent (Ý định):**
       - "search": Tìm mua xe, hỏi giá, hỏi thông tin xe cụ thể.
       - "compare": So sánh 2 hoặc nhiều xe cụ thể (VD: "Vios hay Accent hơn?").
       - "compare_generic": Muốn so sánh nhưng chưa nói xe nào (VD: "So sánh giúp mình").
       - "consult_service": Hỏi về dịch vụ, bảo dưỡng, thủ tục giấy tờ.
       - "chitchat": Chào hỏi, khen chê, nói chuyện phiếm không liên quan xe.

    2. **Smart Filters (Bộ lọc thông minh):**
       - **Price (Giá):** Nếu user nói "tầm 500tr", "dưới 1 tỷ", "1 tỏi 2"... hãy quy đổi ra số nguyên VNĐ.
         -> price_min: int hoặc 0
         -> price_max: int hoặc 0 (Nếu "tầm 500tr" -> min 450tr, max 550tr).
       - **Body Type (Dáng xe):** Map từ khóa:
         "gầm cao" -> ["suv", "mpv", "crossover"]
         "xe gia đình" -> ["mpv", "suv", "sedan"]
         "xe chở hàng", "bán tải" -> ["pickup"]
         "xe nhỏ", "đi phố" -> ["hatchback", "sedan"]
       - **Transmission:** "tự động"/"AT" -> "automatic", "số sàn"/"MT" -> "manual".
       - **Fuel:** "máy dầu" -> "diesel", "máy xăng" -> "petrol", "xe điện" -> "electric".
       - **Usage (Mục đích - Context):** "chạy dịch vụ", "grab" -> "service"; "đi phượt" -> "travel"; "cho vợ đi chợ" -> "daily".
       - **Features (Tính năng):** Trích xuất list các từ khóa: ["sunroof" (cửa sổ trời), "360_camera" (cam 360), "leather" (ghế da), "adas" (an toàn/sensing), "smartkey"].
       - **Performance (Hiệu suất):** Nếu user dùng từ "mạnh mẽ", "bốc", "thể thao", "đạp sướng" -> set "high_performance": true.
       - **Condition (Tình trạng):** "xe lướt", "mới cứng" -> "like_new"; "xe cũ", "giá rẻ" -> "used".
       - **Strictness (Độ khắt khe):** 
         - Nếu user dùng từ: "chỉ mua", "bắt buộc", "phải là" -> set "strict_mode": true.
         - Nếu user nói: "gợi ý", "tham khảo", "tầm tầm" -> set "strict_mode": false.
    3. **Brands:** Trích xuất tên hãng (Toyota, Mazda, Mercedes...) -> lowercase.

    [YÊU CẦU ĐẦU RA]
    Trả về JSON duy nhất, format chuẩn RFC 8259. KHÔNG thêm markdown (```json).
    Format mẫu:
    {{
        "is_car_related": true,
        "intent": "search",
        "mentioned_brands": ["toyota"],
        "filters": {{
            "price_min": 0,
            "price_max": 600000000,
            "min_year": 2018,
            "body_type": ["sedan", "hatchback"],
            "transmission": "automatic",
            "fuel_type": null,
            "min_seats": 0,
            "features": ["sunroof", "adas"],  
            "high_performance": true,         
            "car_condition": "like_new",
            "strict_mode": false       
        }},
        "user_context": {{
            "usage": "family", 
            "priority": "safety" (nếu khách nhắc đến an toàn, bền bỉ...)
        }},
        "reply_suggestion": "Câu trả lời xã giao nếu is_car_related=false"
    }}
    """
    
    try:
        # Cấu hình model để trả về kết quả nhất quán (Deterministic)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", # Nên dùng Flash cho tốc độ và Logic tốt
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, # Nhiệt độ = 0 để trích xuất chính xác tuyệt đối
                top_p=1.0,
            )
        )
        
        raw_text = response.text.strip()
        
        # 🛡️ ROBUST PARSING (Chống lỗi JSON)
        # Sử dụng Regex để tìm block JSON hợp lệ
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            
            # Post-processing (Xử lý hậu kỳ an toàn)
            filters = data.get("filters", {})
            
            # Đảm bảo các field quan trọng luôn tồn tại để Code phía sau không lỗi
            final_data = {
                "is_car_related": data.get("is_car_related", True),
                "intent": data.get("intent", "search"),
                "mentioned_brands": data.get("mentioned_brands", []),
                "filters": {
                    "min_year": filters.get("min_year"),
                    "min_power": filters.get("min_power"),
                    "fuel_type": filters.get("fuel_type"),
                    "min_seats": filters.get("min_seats"),
                    # Các field nâng cấp mới
                    "price_min": filters.get("price_min", 0),
                    "price_max": filters.get("price_max", 0),
                    "features": filters.get("features", []),
                    "high_performance": filters.get("high_performance", False),
                    "car_condition": filters.get("car_condition", "any"),
                    "body_type": filters.get("body_type", []), # List
                    "transmission": filters.get("transmission", "any")
                },
                "user_context": data.get("user_context", {}),
                "reply_suggestion": data.get("reply_suggestion", "")
            }
            
            print(f"🧠 [NLU Analysis] Intent: {final_data['intent']} | Brands: {final_data['mentioned_brands']}")
            if final_data['filters']['price_max']:
                 print(f"   -> Detected Budget: {final_data['filters']['price_min']:,} - {final_data['filters']['price_max']:,} VNĐ")
                 
            return final_data

        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print(f"⚠️ [NLU Error] Phân tích thất bại: {e}")
        # Fallback an toàn tối đa
        return {
            "is_car_related": True, 
            "intent": "search", 
            "mentioned_brands": [], 
            "filters": {}, 
            "user_context": {},
            "reply_suggestion": ""
        }
    
# ==============================================================================
# 2. LOGIC LỌC THÔNG MINH & CAO CẤP (ADVANCED SMART FILTER)
# ==============================================================================

def is_text_similar(a: str, b: str, threshold=0.7):
    """Kiểm tra 2 chuỗi có giống nhau không (chấp nhận lỗi chính tả nhẹ)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def apply_smart_filters(candidates_df, user_profile: UserProfileReq, intent_data: dict):
    """
    🧠 CONTEXT-AWARE SOFT FILTERING
    Nguyên tắc: 
    1. Chat Context > User Profile (Lời nói hiện tại quan trọng nhất).
    2. Soft Penalty: Không xóa xe, chỉ trừ điểm nếu không khớp.
    3. Fallback: Nếu trừ điểm quá tay khiến list rỗng, trả về xe điểm cao nhất dù thấp.
    """
    scored_candidates = []
    
    # 1. Lấy dữ liệu Context (Ưu tiên cao nhất)
    chat_brands = [b.lower() for b in intent_data.get("mentioned_brands", [])]
    extracted_filters = intent_data.get("filters", {}) or {}
    
    # Check chế độ khắt khe (do AI phán đoán)
    is_strict = extracted_filters.get("strict_mode", False)
    
    for _, row in candidates_df.iterrows():
        # Lấy điểm gốc từ Engine (đã tính toán vector tương đồng)
        # Giả sử điểm gốc dao động 60-90
        base_score = float(row.get('match_percent', 70))
        current_score = base_score
        
        car_obj = map_car_to_frontend(row, match_score=base_score)
        car_brand = car_obj['brand'].lower()
        car_price = car_obj['price']
        
        reasons = [] # Ghi lại lý do bị trừ điểm để debug hoặc giải thích

        # ---------------------------------------------------------
        # A. LOGIC HÃNG XE (BRAND) - Priority: Chat > Profile
        # ---------------------------------------------------------
        if chat_brands:
            # User ĐANG hỏi về hãng này -> Kiểm tra kỹ
            match_found = False
            for brand in chat_brands:
                if brand in car_brand or is_text_similar(brand, car_brand):
                    match_found = True
                    break
            
            if match_found:
                current_score += 15 # Cộng điểm mạnh vì đúng ý user ngay lúc này
            else:
                # Sai hãng user đang hỏi
                penalty = 60 if is_strict else 30 # Nếu user "chỉ mua Audi" -> trừ 60, còn "tham khảo" -> trừ 30
                current_score -= penalty
                reasons.append(f"Không phải hãng {chat_brands[0]}")
                
        elif user_profile.preferredBrands:
            # User KHÔNG nói hãng nào trong chat -> Dùng Profile (Ưu tiên thấp hơn)
            if any(pb.lower() in car_brand for pb in user_profile.preferredBrands):
                current_score += 5 # Cộng nhẹ
            # Không trừ điểm nếu không khớp profile (để user khám phá hãng mới)

        # ---------------------------------------------------------
        # B. LOGIC GIÁ TIỀN (PRICE) - Fuzzy Range
        # ---------------------------------------------------------
        # Ưu tiên giá trong chat (context) -> rồi mới tới profile
        target_min = extracted_filters.get("price_min") or user_profile.priceRange[0]
        target_max = extracted_filters.get("price_max") or user_profile.priceRange[1]
        
        # Nếu target_max = 0 hoặc quá lớn (vô lý), bỏ qua check max
        if target_max > 100000000: # > 100tr mới check
            if car_price > target_max:
                # Tính độ lệch giá (Over-budget)
                diff_percent = (car_price - target_max) / target_max
                
                if diff_percent < 0.1: # Lố < 10% (VD: Có 1 tỷ, xe 1 tỷ 1) -> OK
                    current_score -= 5 
                elif diff_percent < 0.3: # Lố < 30% -> Trừ vừa
                    current_score -= 20
                    reasons.append("Vượt ngân sách")
                else: # Lố quá nhiều -> Trừ nặng
                    current_score -= 50
                    reasons.append("Giá quá cao")
            
            elif car_price < target_min * 0.8: # Rẻ hơn quá nhiều (VD: tìm xe sang mà gợi ý xe cỏ)
                current_score -= 10 
                reasons.append("Giá thấp hơn mong đợi")

        # ---------------------------------------------------------
        # C. LOGIC NĂM & CÔNG NGHỆ (Technical Specs)
        # ---------------------------------------------------------
        req_min_year = extracted_filters.get("min_year")
        if req_min_year and car_obj['year'] < req_min_year:
            # Mỗi năm cũ hơn trừ 3 điểm
            diff = req_min_year - car_obj['year']
            current_score -= (diff * 3)
            if diff > 5: reasons.append("Đời xe hơi sâu")

        # ---------------------------------------------------------
        # D. TỔNG KẾT & CHỐT
        # ---------------------------------------------------------
        # Clip điểm (0-100)
        final_score = max(0, min(100, int(current_score)))
        
        car_obj['matchScore'] = final_score
        # Nếu có lý do trừ điểm, update vào matchReason (để hiển thị UI nếu muốn)
        if reasons:
            car_obj['matchReason'] = f"Lưu ý: {', '.join(reasons)}"
        
        # Ngưỡng sàn: Chỉ lấy xe trên 40 điểm
        if final_score >= 40:
            scored_candidates.append(car_obj)

    # Sắp xếp giảm dần theo điểm
    scored_candidates.sort(key=lambda x: x['matchScore'], reverse=True)
    
    # --- FALLBACK THÔNG MINH ---
    # Nếu lọc xong mà rỗng (do trừ điểm quá tay), trả về top 3 xe có điểm cao nhất trong đám bị loại
    # Để tránh việc trả về rỗng hoàn toàn
    if not scored_candidates and candidates_df is not None and len(candidates_df) > 0:
        print("⚠️ Soft filter quá gắt, kích hoạt Rescue Mode.")
        # Lấy lại tất cả, sort và trả về top 3
        backup_list = []
        for _, row in candidates_df.iterrows():
            backup_list.append(map_car_to_frontend(row, match_score=40))
        return backup_list[:3]

    return scored_candidates

# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Luồng xử lý chính:
    1. Map Input FE -> Backend Context
    2. Recommender Engine -> Lấy 50 xe tiềm năng (Retrieval)
    3. API Filters -> Lọc theo Giá, Hãng, Hộp số (Post-Filtering)
    4. LLM Rerank -> Chọn 3 xe tốt nhất & Viết lời thoại (Reranking)
    5. Return -> JSON chuẩn cho FE
    """
    print(f"📩 Chat Request: {req.message}")
    print(f"   Profile: {req.userProfile}")
    # 1. BƯỚC 1: PHÂN TÍCH Ý ĐỊNH (INTENT ANALYSIS)
    intent_data = analyze_user_intent(req.message)
    print(f"🧠 Intent: {intent_data}")

    # 2. XỬ LÝ CASE 1: KHÔNG LIÊN QUAN / TÀO LAO / CHITCHAT
    if not intent_data.get("is_car_related", True) or intent_data.get("intent") == "chitchat":
        return {
            "role": "assistant",
            "content": intent_data.get("reply_suggestion", "Tôi là trợ lý ảo chuyên về xe hơi. Tôi có thể giúp bạn tìm chiếc xe ưng ý không?"),
            "cars": [] # Không trả về xe nào cả -> UI sẽ không hiện thẻ xe lung tung
        }

    # 3. XỬ LÝ CASE 2: SO SÁNH CHUNG CHUNG (COMPARE GENERIC)
    # User: "So sánh đi", "So sánh giúp mình" (Mà không nói xe nào)
    if intent_data.get("intent") == "compare_generic":
        return {
            "role": "assistant",
            "content": "Bạn muốn so sánh những mẫu xe nào? Hãy chọn 'Thêm vào so sánh' trên các thẻ xe, hoặc nói rõ tên 2 dòng xe bạn đang phân vân nhé (Ví dụ: So sánh Vios và Accent).",
            "cars": [] # Không trả về xe
        }
    
    age = req.userProfile.age
    income = req.userProfile.income # USD/năm
    purpose = req.userProfile.purpose
    marital = req.userProfile.maritalStatus

    # Mặc định
    persona = "Family"

    # Logic ưu tiên:
    if income >= 50000000: # Lương cao -> Auto là Boss
        persona = "Boss"
    elif purpose == "commute" and age < 25 and income < 3000000: # Trẻ, lương thấp, đi làm -> Student
        persona = "Student"
    elif purpose == "service": # Chạy dịch vụ -> Cần bền -> Coi như Family/Commute
        persona = "Family"
    elif purpose == "travel": # Đi phượt -> Racer/Family
        persona = "Racer" if age < 30 else "Family"
    
    # ---------------------------------------------------------
    # 2. TẠO PROFILE ĐẦY ĐỦ (Cách 2 nâng cấp)
    # ---------------------------------------------------------
    backend_profile = {
        "persona": persona,
        "age": age,
        "salary": income,
        "is_married": 1 if marital == 'married' else 0,
        "is_rich": True if income >= 50000000 else False,
        "liked_history": user_interactions.get(req.sessionId, [])
    }
    # 4.2. Merge thêm các bộ lọc sâu từ LLM (Năm, Máy, Odo...)
    # 1. Truyền Hãng xe (VD: Audi)
    if intent_data.get("mentioned_brands"):
        backend_profile["specific_brands"] = intent_data["mentioned_brands"]
        print(f"🎯 [Engine] Ưu tiên lọc hãng: {intent_data['mentioned_brands']}")

    # 2. Truyền Bộ lọc chi tiết từ NLU (Năm, Giá,...)
    extracted_filters = intent_data.get("filters", {})
    if extracted_filters:
        # Chỉ lấy các giá trị không null
        clean_filters = {k: v for k, v in extracted_filters.items() if v is not None}
        if clean_filters.get('price_max'):
             backend_profile["max_price_override"] = clean_filters['price_max']
             
        backend_profile.update(clean_filters)

    # 2. Gọi Engine (Lấy dư ra 50 xe để còn lọc lại)
    candidates_df = recsys.recommend(backend_profile, top_k=50)

    if candidates_df.empty:
        return {
            "role": "assistant",
            "content": "Rất tiếc, với các tiêu chí kỹ thuật khắt khe như vậy, tôi chưa tìm thấy chiếc xe nào trong kho dữ liệu. Bạn thử nới lỏng yêu cầu (ví dụ giảm đời xe hoặc công suất) xem sao nhé?",
            "cars": []
        }

    # 4. ÁP DỤNG SMART FILTER (POST-PROCESSING)
    # Bước này lọc lại theo Giá tiền, Hãng (ưu tiên Chat > Profile)
    filtered_cars = apply_smart_filters(candidates_df, req.userProfile, intent_data)
    # filtered_cars = []
    
    # # Duyệt qua kết quả từ Engine
    # for _, row in candidates_df.iterrows():
    #     # Lấy điểm số mà Engine đã tính (bao gồm cả điểm cộng cho hãng/giá nếu có)
    #     score = row.get('match_percent', 85)
        
    #     # Chuyển đổi sang format JSON cho Frontend
    #     car_obj = map_car_to_frontend(row, match_score=score)
        
    #     # Nếu muốn, bạn có thể cập nhật matchReason cơ bản ở đây
    #     if intent_data.get("mentioned_brands"):
    #          # Nếu user hỏi hãng, và xe này đúng hãng -> note lại
    #          requested_brands = [b.lower() for b in intent_data["mentioned_brands"]]
    #          if car_obj['brand'].lower() in requested_brands:
    #              car_obj['matchReason'] = "Đúng thương hiệu bạn tìm"
        
    #     filtered_cars.append(car_obj)

    # print(f"🚀 [Pipeline] Engine trả về {len(filtered_cars)} xe -> Chuyển thẳng cho LLM Rerank.")
    
    final_cars = []
    final_content = ""
    message_prefix = ""

    # 6. RERANKING & RESPONSE GENERATION (Chia nhánh Search vs Compare)
    
    # NHÁNH A: SO SÁNH (COMPARE)
    if intent_data.get("intent") == "compare" and len(filtered_cars) >= 2:
        # Lấy tối đa 4 xe để user so sánh
        final_cars = filtered_cars[:4]
        
        # Nhờ Gemini viết đoạn so sánh ngắn
        car_names = ", ".join([c['name'] for c in final_cars])
        prompt = f"Khách hỏi: '{req.message}'. Tôi tìm được: {car_names}. Hãy viết đoạn ngắn (dưới 50 từ) mời khách bấm vào nút So sánh trên các thẻ xe."
        
        try:
            res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            final_content = message_prefix + res.text
        except:
            final_content = message_prefix + "Dưới đây là các xe bạn yêu cầu. Hãy chọn 'Thêm vào so sánh' để xem chi tiết."

    # NHÁNH B: TÌM KIẾM (SEARCH) - Mặc định
    else:
        # --- BƯỚC 1: SYSTEM SELECTION (Hệ thống tự chọn) ---
        # Sắp xếp danh sách xe từ Engine theo điểm số matchScore (cao -> thấp)
        # filtered_cars là danh sách 50 xe từ Engine trả về
        filtered_cars.sort(key=lambda x: x['matchScore'], reverse=True)
        
        # Cắt lấy Top 3 xe xuất sắc nhất
        final_cars = filtered_cars[:3]
        
        # Nếu không có xe nào (Fallback)
        if not final_cars:
            print("⚠️ Filter quá chặt. Dùng Fallback.")
            fallback_df = recsys.df_cars.sample(3) 
            final_cars = [map_car_to_frontend(row, match_score=60) for _, row in fallback_df.iterrows()]
            message_prefix = "Hiện chưa tìm thấy xe chính xác theo yêu cầu, nhưng bạn có thể tham khảo: "

        # --- BƯỚC 2: AI EXPLANATION ---
        # Chỉ gọi AI để viết lời thoại cho 3 xe đã chốt
        ai_response = llm_rerank_and_explain(req.message, req.userProfile, final_cars)
        
        # Ghép lời thoại
        final_content = message_prefix + ai_response.get("analysis", "Đây là các gợi ý phù hợp nhất.")

    # 7. TRẢ KẾT QUẢ
    return {
        "role": "assistant",
        "content": final_content,
        "cars": final_cars
    }

@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    API nhận Feedback từ Frontend.
    Sử dụng BackgroundTasks để không block request của user.
    """
    # 1. Validation cơ bản (nếu cần)
    if not req.car_id or not req.user_id:
        return {"status": "error", "message": "Missing info"}

    print(f"👍 Feedback nhận được: User {req.user_id} - {req.action} - Xe {req.car_id}")
    
    # 2. Đẩy việc ghi file vào nền (Chạy song song, trả response ngay lập tức)
    background_tasks.add_task(log_feedback_to_csv, req.user_id, req.car_id, req.action)
    
    # 3. (Tuỳ chọn nâng cao) Real-time Update
    # Nếu hệ thống cực xịn, tại đây có thể gọi hàm update weight cho model
    # Nhưng với đồ án, việc lưu log để train sau là đủ chuẩn.

    return {
        "status": "success", 
        "message": "Feedback recorded successfully",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/similar/{car_id}")
def similar_cars_endpoint(car_id: str): # Đổi thành str để nhận mọi loại ID
    """
    Endpoint lấy xe tương tự (Hybrid Approach).
    Kết hợp sức mạnh của Matrix Factorization và Content Filtering.
    """
    try:
        # Gọi hàm Hybrid mới
        similar_df = recsys.get_similar_cars_item_based(car_id, top_k=4)
        
        cars = []
        for _, row in similar_df.iterrows():
            # Xe từ CF thường có độ tin cậy cao hơn Content
            score = 90 if 'sim_score' not in row else int(row['sim_score']) # sim_score từ content-based logic
            
            # Clip score
            score = max(70, min(99, score))
            
            mapped_car = map_car_to_frontend(row, match_score=score)
            
            # Cập nhật lý do
            if 'sim_score' in row:
                mapped_car['matchReason'] = "Tương đồng về thông số kỹ thuật & tầm giá"
            else:
                mapped_car['matchReason'] = "Được nhiều người cùng sở thích quan tâm"
                
            cars.append(mapped_car)
            
        return cars
        
    except Exception as e:
        print(f"❌ Error getting similar cars: {e}")
        return []

@app.get("/api/cars")
def get_all_cars_endpoint():
    """
    API trả về toàn bộ danh sách xe hiện có trong kho dữ liệu (scraped_cars.csv).
    Phục vụ cho trang Showroom để hiển thị lưới sản phẩm.
    """
    if recsys is None or recsys.df_cars is None:
        return []

    try:
        all_cars = []
        # Duyệt qua toàn bộ DataFrame xe
        # Lưu ý: Nếu dữ liệu > 10.000 xe, nên làm phân trang (pagination) ở backend.
        # Với dữ liệu đồ án (< 2000 xe), trả về hết list là OK.
        for _, row in recsys.df_cars.iterrows():
            # Sử dụng lại hàm map_car_to_frontend để đảm bảo cấu trúc JSON đồng nhất với phần Chat
            # match_score = 0 vì đây là danh sách thô, không phải gợi ý cá nhân hóa
            car_obj = map_car_to_frontend(row, match_score=0)
            
            # Ghi đè matchReason mặc định cho trang showroom
            car_obj['matchReason'] = "Sẵn sàng giao ngay" 
            
            all_cars.append(car_obj)

        print(f"📦 [API] Showroom: Đã trả về {len(all_cars)} xe.")
        return all_cars

    except Exception as e:
        print(f"❌ Lỗi lấy danh sách xe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks):
    """
    API để admin kích hoạt học lại từ feedback.
    Chạy ngầm (Background) để không treo server.
    """
    background_tasks.add_task(recsys.retrain_model)
    return {"status": "success", "message": "Đang huấn luyện lại model trong nền..."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)