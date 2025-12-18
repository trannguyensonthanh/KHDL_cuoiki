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
    💎 ULTRA-PREMIUM LLM RERANKER
    Chức năng:
    1. Phân tích sâu tâm lý người dùng (Psychological Profiling).
    2. So khớp đa chiều (Multidimensional Matching): Giá, Tech, Brand, Nhu cầu ngầm.
    3. Chọn ra 3 xe tốt nhất ("Golden Trio").
    4. Viết lời tư vấn bán hàng thuyết phục (Persuasive Copywriting).
    """
    
    # 1. CHUẨN BỊ DỮ LIỆU ĐẦU VÀO GIÀU NGỮ CẢNH (RICH CONTEXT)
    cars_context = ""
    for i, car in enumerate(car_list):
        # Lấy tối đa 5 tính năng nổi bật để tránh quá tải token
        feats = ", ".join(car.get('features', [])[:5]) if car.get('features') else "Cơ bản"
        specs = car.get('specs', {})
        
        cars_context += (
            f"--- CAR ID: {i} ---\n"
            f"Model: {car['name']} ({car['year']}) | Hãng: {car['brand']}\n"
            f"Giá: {car['price']:,} VNĐ | ODO/Mới: {specs.get('fuelConsumption', 'N/A')}\n"
            f"Thông số: {car['seats']} chỗ, {car['transmission']}, {car['fuelType']}, {specs.get('horsepower', 0)}HP\n"
            f"Tính năng: {feats}\n"
        )

    # 2. XÂY DỰNG PROMPT KỸ THUẬT CAO (CHAIN-OF-THOUGHT PROMPT)
    # Xác định giọng điệu dựa trên profile (Dynamic Persona Adaptation)
    tone_instruction = "Chuyên nghiệp, tin cậy và khách quan."
    if user_profile.income > 25000000 or user_profile.age > 45:
        tone_instruction = "Sang trọng, lịch thiệp, tôn trọng đẳng cấp khách hàng (gọi là 'quý khách')."
    elif user_profile.age < 30:
        tone_instruction = "Trẻ trung, năng động, tập trung vào công nghệ và tốc độ."
    elif user_profile.purpose == "family":
        tone_instruction = "Ấm áp, quan tâm, nhấn mạnh sự an toàn và tiện nghi cho gia đình."

    prompt = f"""
    [VAI TRÒ]
    Bạn là một chuyên gia tư vấn xe hơi cao cấp (AI Concierge) với 20 năm kinh nghiệm. 
    Nhiệm vụ của bạn là chọn ra chính xác 3 chiếc xe phù hợp nhất cho khách hàng từ danh sách ứng viên và thuyết phục họ.

    [HỒ SƠ KHÁCH HÀNG]
    - Tuổi: {user_profile.age} | Thu nhập: {user_profile.income} USD/năm
    - Tình trạng hôn nhân: {user_profile.maritalStatus} | Mục đích: {user_profile.purpose}
    - Câu hỏi/Nhu cầu hiện tại: "{user_msg}"

    [DANH SÁCH ỨNG VIÊN]
    {cars_context}

    [QUY TRÌNH TƯ DUY - CHAIN OF THOUGHT]
    1. Phân tích ý định ngầm (Intent Detection): Khách quan tâm giá rẻ, sĩ diện, an toàn hay cảm giác lái?
    2. Lọc kỹ thuật: Loại bỏ xe quá ngân sách hoặc sai nhu cầu (ví dụ hỏi xe 7 chỗ mà list có xe 4 chỗ).
    3. Chọn lọc: Chọn 3 xe tốt nhất (Best Value, Best Fit, Best Experience).
    4. Soạn thảo lời thoại: Viết lời khuyên ngắn gọn nhưng "chạm" vào tử huyệt cảm xúc của khách.

    [YÊU CẦU ĐẦU RA]
    Trả về định dạng JSON chuẩn (RFC 8259), không có Markdown, không giải thích ngoài JSON:
    {{
        "selected_indices": [index_xe_1, index_xe_2, index_xe_3],
        "analysis": "Lời tư vấn dưới 70 từ. Xưng 'tôi'. {tone_instruction} Hãy nhắc khéo đến tính năng cụ thể của xe được chọn để tăng tính thuyết phục."
    }}
    """

    # 3. GỌI API GEMINI VỚI CẤU HÌNH TỐI ƯU
    try:
        # Sử dụng model thông minh nhất bạn có quyền truy cập (Ưu tiên Flash hoặc Pro 1.5)
        # Nếu đang dùng genai SDK mới:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", # Hoặc "gemini-2.5-flash" nếu bạn bị giới hạn
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4, # Giảm nhiệt độ để logic chính xác hơn
                top_p=0.8,
            )
        )
        
        raw_text = response.text.strip()
        
        # 4. XỬ LÝ LỖI PARSING JSON MẠNH MẼ (ROBUST PARSING)
        # Tìm chuỗi JSON hợp lệ giữa dấu { và } cuối cùng
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            # Validation: Đảm bảo trả về đủ 3 chỉ số (hoặc ít hơn nếu list gốc ít xe)
            indices = result.get("selected_indices", [])
            valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(car_list)]
            
            # Nếu LLM trả về rỗng hoặc sai index, fallback lấy 3 xe đầu
            if not valid_indices:
                valid_indices = [0, 1, 2][:len(car_list)]
            
            # Cập nhật lại kết quả đã validate
            result["selected_indices"] = valid_indices
            print(f"✅ [LLM Rerank] Selected: {valid_indices} | Reason: {result.get('analysis')[:50]}...")
            return result
        else:
            raise ValueError("No JSON found in LLM response")

    except Exception as e:
        print(f"⚠️ [Rerank Error] Lỗi xử lý AI: {e}")
        print(f"   -> Raw response: {locals().get('raw_text', 'N/A')}")
        
        # 5. FALLBACK THÔNG MINH (RULE-BASED FALLBACK)
        # Nếu AI tạch, dùng logic Python để chọn xe tốt nhất thay vì random
        # Ví dụ: Sắp xếp theo matchScore có sẵn
        sorted_indices = sorted(range(len(car_list)), key=lambda k: car_list[k].get('matchScore', 0), reverse=True)
        return {
            "selected_indices": sorted_indices[:3],
            "analysis": f"Hệ thống AI đang bận, nhưng dựa trên dữ liệu kỹ thuật, đây là 3 lựa chọn khớp nhất với nhu cầu '{user_profile.purpose}' của bạn."
        }
    
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
            "car_condition": "like_new"       
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
    Bộ lọc hậu kỳ "Gatekeeper": Đảm bảo xe trả về phải cực kỳ sát với nhu cầu.
    Nâng cấp: Fuzzy matching, Price tolerance, Deep specs checking.
    """
    filtered_cars_json = []
    
    # 1. Thu thập Context từ Chat (Quan trọng nhất)
    mentioned_brands = [b.lower() for b in intent_data.get("mentioned_brands", [])]
    extracted_filters = intent_data.get("filters", {}) or {} # Các filter LLM trích xuất (năm, chỗ, nhiên liệu...)
    
    for _, row in candidates_df.iterrows():
        # Lấy điểm gốc từ Engine
        base_score = row.get('match_percent', 85)
        car_obj = map_car_to_frontend(row, match_score=base_score)
        
        is_valid = True
        reject_reason = "" # Debug lý do loại bỏ (nếu cần log)

        # ---------------------------------------------------------
        # A. LOGIC HÃNG XE (BRAND) - Có Fuzzy Matching
        # ---------------------------------------------------------
        car_brand_clean = car_obj['brand'].lower()
        
        if mentioned_brands:
            # Ưu tiên 1: User vừa nhắc tên hãng trong chat -> Bắt buộc phải đúng hãng đó
            # Dùng fuzzy match: "mec" khớp "mercedes", "toyta" khớp "toyota"
            match_found = False
            for brand in mentioned_brands:
                if brand in car_brand_clean or is_text_similar(brand, car_brand_clean):
                    match_found = True
                    break
            if not match_found:
                is_valid = False
                reject_reason = "Wrong Brand (Context)"
                
        elif user_profile.preferredBrands and len(user_profile.preferredBrands) > 0:
            # Ưu tiên 2: Profile User (nếu không nhắc hãng trong chat)
            match_found = False
            for fav in user_profile.preferredBrands:
                if fav.lower() in car_brand_clean:
                    match_found = True
                    break
            if not match_found:
                is_valid = False
                reject_reason = "Wrong Brand (Profile)"

    # [NEW] Logic Lọc Giá Thông Minh từ NLU
    extracted_filters = intent_data.get("filters", {})
    price_max = extracted_filters.get("price_max", 0)
    price_min = extracted_filters.get("price_min", 0)

    # Nếu NLU phát hiện ra giá trong chat -> Ghi đè lên Profile User
    if price_max > 0:
        # Logic: Giá xe phải nằm trong vùng user nói
        # Cho phép dung sai 5%
        if car_obj['price'] > price_max * 1.05 or car_obj['price'] < price_min * 0.95:
            is_valid = False
            reject_reason = "Price mismatch (Chat Context)"

    # [NEW] Logic Body Type (Gầm cao/Thấp)
    req_body_types = extracted_filters.get("body_type", []) # List ['suv', 'sedan'...]
    if is_valid and req_body_types:
        # Cần logic map từ CSV sang body type (Giả sử bạn đã có hàm classify_car_type ở engine)
        # Ở đây so sánh string đơn giản
        car_type_guess = "sedan" # Default
        if car_obj['seats'] >= 7: car_type_guess = "mpv"
        elif "suv" in car_obj['name'].lower(): car_type_guess = "suv"
        
        # Check if car matches any requested type
        # (Phần này nên làm kỹ hơn ở Engine, nhưng lọc sơ ở đây cũng tốt)
        pass 
        # ---------------------------------------------------------
        # B. LOGIC GIÁ TIỀN (PRICE) - Có Tolerance (Dung sai)
        # ---------------------------------------------------------
        # Nếu đang so sánh, bỏ qua giá để user thấy sự khác biệt
        if is_valid and intent_data.get('intent') != 'compare':
            if user_profile.priceRange and len(user_profile.priceRange) == 2:
                min_p, max_p = user_profile.priceRange
                car_price = car_obj['price']
                
                # TOLERANCE 10%: Cho phép giá cao hơn ngân sách 10% nếu xe ngon
                # Ví dụ: Tìm xe 1 tỷ, xe 1 tỷ 50tr vẫn chấp nhận
                upper_limit = max_p * 1.1 if max_p > 0 else float('inf')
                lower_limit = min_p * 0.9 # Thấp hơn 10% vẫn ok
                
                if max_p > 0 and not (lower_limit <= car_price <= upper_limit):
                    is_valid = False
                    reject_reason = "Price out of range"

        # ---------------------------------------------------------
        # C. LOGIC KỸ THUẬT SÂU (DEEP SPECS CHECK) - Từ LLM trích xuất
        # ---------------------------------------------------------
        if is_valid and extracted_filters:
            # 1. Năm sản xuất (Min Year)
            if extracted_filters.get('min_year') and car_obj['year'] < extracted_filters['min_year']:
                is_valid = False
            
            # 2. Nhiên liệu (Fuel Type)
            if is_valid and extracted_filters.get('fuel_type'):
                req_fuel = extracted_filters['fuel_type'].lower() # 'xăng', 'dầu', 'điện'
                car_fuel = car_obj['fuelType'].lower()
                
                # Map tương đối: 'petrol' khớp 'xăng', 'diesel' khớp 'dầu'
                fuel_map = {'petrol': 'xăng', 'diesel': 'dầu', 'electric': 'điện', 'ev': 'điện'}
                req_fuel_norm = fuel_map.get(req_fuel, req_fuel)
                
                if req_fuel_norm not in car_fuel:
                    is_valid = False

            # 3. Số chỗ (Seats) - Ví dụ user chat "Tìm xe 7 chỗ"
            # (Giả sử bạn đã update analyze_user_intent để trích xuất min_seats)
            if is_valid and extracted_filters.get('min_seats'): 
                 if car_obj['seats'] < extracted_filters['min_seats']:
                     is_valid = False

        # ---------------------------------------------------------
        # D. LOGIC HỘP SỐ (TRANSMISSION)
        # ---------------------------------------------------------
        if is_valid and user_profile.transmission and user_profile.transmission != 'any':
            req_trans = user_profile.transmission # 'manual' / 'automatic'
            car_trans_str = str(car_obj['transmission']).lower()
            
            is_auto_car = 'tự động' in car_trans_str or 'at' in car_trans_str or 'cvt' in car_trans_str
            
            if req_trans == 'automatic' and not is_auto_car:
                is_valid = False
            elif req_trans == 'manual' and is_auto_car:
                is_valid = False

        # ---------------------------------------------------------
        # E. LOGIC TỪ KHÓA TÍNH NĂNG (KEYWORD MATCHING)
        # ---------------------------------------------------------
        # Nếu user chat "xe có cửa sổ trời", kiểm tra trong features
        if is_valid and 'search_query' in intent_data: 
            # (Lưu ý: Bạn cần pass nguyên câu query vào intent_data hoặc lấy từ req)
            pass 
            # Phần này thường Engine đã làm ở bước Retrieval, 
            # ở đây ta chỉ lọc nếu muốn cực kỳ nghiêm ngặt.
            
        if is_valid:
            filtered_cars_json.append(car_obj)
            
    return filtered_cars_json

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
        # Các trường này giúp Engine (nếu được nâng cấp) lọc tốt hơn
        "is_rich": True if income >= 50000000 else False,
        # --- [BỔ SUNG] TRUYỀN LỊCH SỬ LIKE VÀO ENGINE ---
        # Lấy session ID hoặc User ID từ request (giả sử req.sessionId hoặc req.userProfile.userId)
        # Ở đây mình dùng logic user_interactions global dict đã có sẵn trong api.py
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
        
        # Map price_max từ NLU sang priceRange của Engine nếu có
        # Engine dùng 'price_code' hoặc lọc thủ công, nhưng ta có thể pass tham số để Engine xử lý
        if clean_filters.get('price_max'):
             # Ghi đè logic giá của Engine nếu user nói rõ ngân sách
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

    # # 4. ÁP DỤNG SMART FILTER (POST-PROCESSING)
    # # Bước này lọc lại theo Giá tiền, Hãng (ưu tiên Chat > Profile)
    # filtered_cars = apply_smart_filters(candidates_df, req.userProfile, intent_data)
    filtered_cars = []
    
    # Duyệt qua kết quả từ Engine
    for _, row in candidates_df.iterrows():
        # Lấy điểm số mà Engine đã tính (bao gồm cả điểm cộng cho hãng/giá nếu có)
        score = row.get('match_percent', 85)
        
        # Chuyển đổi sang format JSON cho Frontend
        car_obj = map_car_to_frontend(row, match_score=score)
        
        # Nếu muốn, bạn có thể cập nhật matchReason cơ bản ở đây
        if intent_data.get("mentioned_brands"):
             # Nếu user hỏi hãng, và xe này đúng hãng -> note lại
             requested_brands = [b.lower() for b in intent_data["mentioned_brands"]]
             if car_obj['brand'].lower() in requested_brands:
                 car_obj['matchReason'] = "Đúng thương hiệu bạn tìm"
        
        filtered_cars.append(car_obj)

    print(f"🚀 [Pipeline] Engine trả về {len(filtered_cars)} xe -> Chuyển thẳng cho LLM Rerank.")
    
    final_cars = []
    final_content = ""
    message_prefix = ""

    # 5. XỬ LÝ FALLBACK (Nếu lọc xong hết sạch xe)
    if not filtered_cars:
        print("⚠️ Filter quá chặt. Dùng Fallback (Top Trending).")
        # Lấy random 3 xe từ kho làm gợi ý
        fallback_df = recsys.df_cars.sample(3) 
        filtered_cars = [map_car_to_frontend(row, match_score=75) for _, row in fallback_df.iterrows()]
        message_prefix = "Không tìm thấy xe khớp hoàn toàn yêu cầu của bạn, nhưng bạn có thể tham khảo các mẫu xe này: "

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
        # Dùng LLM chọn ra 3 xe tốt nhất (Rerank)
        # Lưu ý: Hàm llm_rerank (hoặc llm_rerank_and_explain) phải được định nghĩa ở trên
        rerank_result = llm_rerank_and_explain(req.message, req.userProfile, filtered_cars)
        
        selected_indices = rerank_result.get("selected_indices", [0, 1, 2])
        for idx in selected_indices:
            if idx < len(filtered_cars):
                final_cars.append(filtered_cars[idx])
        
        # Fallback nếu LLM lỗi
        if not final_cars:
            final_cars = filtered_cars[:3]
            
        final_content = message_prefix + rerank_result.get("analysis", "Đây là những lựa chọn tốt nhất cho bạn.")

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
def similar_cars_endpoint(car_id: int):
    """
    Endpoint Item-Item CF (Kiến thức Slide)
    Khi user bấm vào xem chi tiết 1 xe -> Gọi API này để lấy xe tương tự
    """
    similar_df = recsys.get_similar_cars_item_based(car_id, top_k=3)
    cars = []
    for _, row in similar_df.iterrows():
        cars.append(map_car_to_frontend(row, match_score=0.85)) # Score giả định cao
    return cars

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