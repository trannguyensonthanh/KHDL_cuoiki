import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random
import hashlib
import csv
class BonBanhAutoCrawler:
    def __init__(self):
        self.base_url = "https://bonbanh.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'vi-VN,vi;q=0.9'
        }
        self.data_store = []

    # --- 1. CÁC HÀM XỬ LÝ & CHUẨN HÓA DỮ LIỆU (QUAN TRỌNG) ---
    
    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip() if text else ""

    def parse_price(self, price_str):
        """
        Xử lý các trường hợp:
        - "1 Tỷ 200 Triệu" -> 1200000000
        - "950 Triệu" -> 950000000
        - "Liên hệ" -> 0
        """
        if not price_str: return 0
        
        # Làm sạch chuỗi
        text = price_str.lower().strip()
        
        # Nếu là giá liên hệ/thỏa thuận -> Bỏ qua hoặc để 0
        if 'liên hệ' in text or 'thỏa thuận' in text:
            return 0
            
        total = 0
        
        # 1. Xử lý phần TỶ
        # Regex tìm số đứng trước chữ "tỷ" (chấp nhận dấu chấm hoặc phẩy là thập phân: 1.5 Tỷ)
        ty_match = re.search(r'([\d\.,]+)\s*tỷ', text)
        if ty_match:
            num_str = ty_match.group(1).replace(',', '.') # Chuẩn hóa về dấu chấm float
            try:
                total += float(num_str) * 1_000_000_000
            except: pass
            
        # 2. Xử lý phần TRIỆU
        # Nếu đã có Tỷ, chỉ lấy phần triệu sau chữ tỷ. Nếu chưa có Tỷ, lấy toàn bộ.
        remaining_text = text.split('tỷ')[1] if 'tỷ' in text else text
        
        tr_match = re.search(r'([\d\.,]+)\s*(triệu|tr)', remaining_text)
        if tr_match:
            num_str = tr_match.group(1).replace(',', '.')
            try:
                total += float(num_str) * 1_000_000
            except: pass

        # 3. Trường hợp chỉ có số (ít gặp trên bonbanh nhưng phòng hờ)
        if total == 0:
            # Tìm tất cả số, bỏ qua dấu chấm phân cách hàng nghìn
            clean_digits = re.sub(r'[^\d]', '', text)
            if clean_digits:
                total = int(clean_digits)
                # Nếu số quá nhỏ (< 10.000), có thể là 1.5 (Tỷ) mà regex trên miss, hoặc lỗi
                # Nhưng logic trên bonbanh thường kèm đơn vị tiền tệ nên logic này là fallback.

        return int(total)
    
    def normalize_transmission(self, text):
        """Chuyển đổi về chuẩn Frontend: 'Số sàn' | 'Số tự động'"""
        text = text.lower()
        if 'tự động' in text or 'tự dộng' in text: return 'Số tự động'
        return 'Số sàn'

    def normalize_fuel(self, text):
        """Chuyển đổi về chuẩn Frontend: 'Xăng' | 'Dầu' | 'Điện' | 'Hybrid'"""
        text = text.lower()
        if 'điện' in text: return 'Điện'
        if 'hybrid' in text or 'lai' in text: return 'Hybrid'
        if 'dầu' in text or 'diesel' in text: return 'Dầu'
        return 'Xăng' # Mặc định

    def parse_seats(self, text):
        """Lấy số từ chuỗi '5 chỗ'"""
        try:
            return int(re.search(r'\d+', text).group())
        except:
            return 5 # Mặc định

    def generate_missing_specs(self, car_name, engine_txt, fuel_type):
        """
        🔥 TỰ SINH DỮ LIỆU THIẾU (Mã lực, Torque, Kích thước...)
        Dựa trên tên xe và động cơ để fake số liệu hợp lý.
        """
        # 1. Đoán dung tích động cơ từ text (VD: 2.0L -> 2.0)
        displacement = 1.5 # Mặc định
        match = re.search(r'(\d\.\d)', str(engine_txt))
        if match:
            displacement = float(match.group(1))
        
        # 2. Sinh Mã lực (Horsepower) & Torque giả lập theo dung tích
        # Công thức ước lượng: HP ~= Dung tích * 70-100
        hp_base = int(displacement * 85) + random.randint(-10, 20)
        torque_base = int(hp_base * 1.2) + random.randint(-10, 20)
        
        # Xe điện/Hybrid thì mạnh hơn
        if fuel_type in ['Điện', 'Hybrid']:
            hp_base = int(hp_base * 1.5)
            torque_base = int(torque_base * 1.8)

        # 3. Sinh Tiêu hao nhiên liệu
        fuel_cons = f"{random.uniform(5.5, 9.5):.1f}L/100km"
        if fuel_type == 'Điện': fuel_cons = "0L/100km"
        elif displacement > 2.5: fuel_cons = f"{random.uniform(10.0, 14.0):.1f}L/100km"

        # 4. Sinh Kích thước & Trọng lượng (Dựa vào tên xe có chữ SUV hay không)
        name_lower = car_name.lower()
        if any(x in name_lower for x in ['suv', 'cr-v', 'cx-', 'fortuner', 'everest', 'glc', 'x5']):
            dims = "4700 x 1860 x 1700 mm"
            weight = f"{random.randint(1700, 2200)} kg"
        elif 'morning' in name_lower or 'i10' in name_lower or 'fadil' in name_lower:
            dims = "3600 x 1600 x 1490 mm"
            weight = f"{random.randint(900, 1100)} kg"
        else: # Sedan
            dims = "4600 x 1800 x 1450 mm"
            weight = f"{random.randint(1300, 1600)} kg"

        return {
            "horsepower": hp_base,
            "torque": f"{torque_base} Nm",
            "fuelConsumption": fuel_cons,
            "dimensions": dims,
            "weight": weight
        }

    # --- 2. LOGIC CRAWL ---

    def make_request(self, url):
        try:
            time.sleep(random.uniform(0.5, 1.5)) # Sleep nhẹ
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 200:
                return BeautifulSoup(resp.content, 'html.parser')
        except: pass
        return None

    def get_brands(self):
        print("📡 Đang lấy danh sách hãng...")
        soup = self.make_request(self.base_url)
        brands = []
        if soup:
            nav = soup.find('ul', id='primary-nav')
            if nav:
                for li in nav.find_all('li', class_='menuparent'):
                    tag = li.find(['a', 'span'], class_='mtop-item')
                    if tag:
                        link = tag.get('href') or tag.get('url')
                        if link:
                            full = f"{self.base_url}/{link}" if not link.startswith('http') else link
                            brands.append({'name': self.clean_text(tag.text), 'url': full})
        return brands

    def get_cars(self, brand_url, limit=5):
        soup = self.make_request(brand_url)
        links = []
        if soup:
            # Selector xe của Bonbanh
            items = soup.select('li.car-item a[itemprop="url"]')
            for item in items[:limit]:
                l = item.get('href')
                if l: links.append(f"{self.base_url}/{l}" if not l.startswith('http') else l)
        return links

    def scrape_detail(self, url):
        soup = self.make_request(url)
        if not soup: return None

        try:
            # ID
            id_match = re.search(r'-(\d+)$', url)
            car_id = id_match.group(1) if id_match else hashlib.md5(url.encode()).hexdigest()[:8]

            # Title & Price
            title_div = soup.find('div', class_='title')
            full_title = self.clean_text(title_div.find('h1').text) if title_div else "Xe không tên"
            
            # Tách giá từ tiêu đề (VD: Xe VinFast VF3... - 229 Triệu)
            name, price = full_title, 0
            if '-' in full_title:
                parts = full_title.rsplit('-', 1)
                name = parts[0].strip().replace('Xe ', '') # Bỏ chữ Xe cho gọn
                price = self.parse_price(parts[1])

            # Specs extraction
            specs_raw = {}
            rows = soup.select('.box_car_detail .row') + soup.select('.box_car_detail .row_last')
            for row in rows:
                lbl = row.find('label')
                val = row.find('span', class_='inp')
                if lbl and val:
                    k = self.clean_text(lbl.text).replace(':', '')
                    v = self.clean_text(val.text)
                    specs_raw[k] = v

            # Map fields
            brand = "Khác"
            bc = soup.select('.breadcrum a span strong')
            if len(bc) >= 1: brand = self.clean_text(bc[0].text)

            year = int(specs_raw.get('Năm sản xuất', 2020))
            
            # Chuẩn hóa dữ liệu thô
            trans_norm = self.normalize_transmission(specs_raw.get('Hộp số', ''))
            fuel_norm = self.normalize_fuel(specs_raw.get('Nhiên liệu', '') or specs_raw.get('Động cơ', ''))
            seats_num = self.parse_seats(specs_raw.get('Số chỗ ngồi', '5 chỗ'))
            engine_txt = specs_raw.get('Động cơ', '2.0L')

            # 🔥 SINH DỮ LIỆU THIẾU (AI LOGIC)
            generated_specs = self.generate_missing_specs(name, engine_txt, fuel_norm)

            # Ảnh
            img = soup.find('img', id='img1')
            image_url = img.get('src') if img else "https://placehold.co/600x400?text=No+Image"

            # Features
            des_div = soup.find('div', class_='des_txt')
            desc = self.clean_text(des_div.text) if des_div else ""
            
            feats = []
            keywords = ['ABS', 'EBD', 'Cửa sổ trời', 'Ghế da', 'Camera 360', 'Cảm biến', 'Apple CarPlay', 'Cruise Control', 'Túi khí', 'Start/Stop']
            for k in keywords:
                if k.lower() in desc.lower(): feats.append(k)
            
            # Nếu ít feature quá thì random thêm cho đẹp UI
            if len(feats) < 3:
                feats += random.sample(['Kết nối Bluetooth', 'Màn hình Android', 'Dán phim cách nhiệt', 'Lốp mới'], 2)

            # --- CẤU TRÚC JSON KHỚP 100% VỚI TYPESCRIPT INTERFACE ---
            return {
                "id": str(car_id),
                "name": name,
                "brand": brand,
                "year": year,
                "price": price,
                "image": image_url,
                "seats": seats_num,
                "transmission": trans_norm,
                "fuelType": fuel_norm,
                # Các trường runtime (mặc định)
                "matchScore": 0,
                "matchReason": "",
                # Specs Object (Đã gộp thật + giả)
                "specs": {
                    "engine": engine_txt,
                    "horsepower": generated_specs['horsepower'],
                    "torque": generated_specs['torque'],
                    "fuelConsumption": generated_specs['fuelConsumption'],
                    "dimensions": generated_specs['dimensions'],
                    "weight": generated_specs['weight']
                },
                "description": desc[:300] + "..." if len(desc) > 300 else desc,
                "features": list(set(feats)) # Remove duplicates
            }

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return None

    # --- SỬA LẠI HÀM RUN ĐỂ LƯU CSV ---
    def run(self):
        MAX_BRANDS = 100 
        CARS_PER_BRAND = 25 
        
        brands = self.get_brands()
        print(f"🔥 Tìm thấy {len(brands)} hãng xe. Bắt đầu quét toàn bộ...")

        total_scraped = 0

        for brand in brands[:MAX_BRANDS]: 
            print(f"\n🚙 Đang quét: {brand['name'].upper()}")
            urls = self.get_cars(brand['url'], limit=CARS_PER_BRAND)
            print(f"   Tìm thấy {len(urls)} xe trong hãng {brand['name']}.")
            
            for u in urls:
                time.sleep(random.uniform(1, 2)) 
                data = self.scrape_detail(u)
                if data:
                    self.data_store.append(data)
                    total_scraped += 1
                    print(f"   [{total_scraped}] ✅ {data['name']}")
                
                # Lưu nháp sau mỗi 25 xe
                if total_scraped % 25 == 0:
                    self.save_to_csv() # <--- GỌI HÀM LƯU CSV

        self.save_to_csv()
        print(f"\n🎉 HOÀN TẤT! Đã lưu {total_scraped} xe vào file CSV.")

    # --- HÀM MỚI: LƯU CSV ---
    def save_to_csv(self):
        filename = 'scraped_cars.csv'
        
        # Định nghĩa các cột (Header) cho file CSV
        # Chúng ta tách specs ra thành từng cột riêng để dễ train model sau này
        fieldnames = [
            'id', 'name', 'brand', 'year', 'price', 
            'seats', 'transmission', 'fuelType', 
            'image', 'description', 'features', 
            # Các cột Specs được làm phẳng
            'engine', 'horsepower', 'torque', 'fuelConsumption', 'dimensions', 'weight'
        ]

        try:
            # encoding='utf-8-sig' để mở bằng Excel không bị lỗi phông chữ tiếng Việt
            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for car in self.data_store:
                    # Làm phẳng dữ liệu (Flattening)
                    flat_row = {
                        'id': car['id'],
                        'name': car['name'],
                        'brand': car['brand'],
                        'year': car['year'],
                        'price': car['price'],
                        'seats': car['seats'],
                        'transmission': car['transmission'],
                        'fuelType': car['fuelType'],
                        'image': car['image'],
                        'description': car['description'].replace('\n', ' '), # Xóa xuống dòng thừa
                        # Chuyển list features thành chuỗi: "ABS, Túi khí, Camera"
                        'features': ", ".join(car['features']),
                        # Lấy thông số từ object specs
                        'engine': car['specs'].get('engine'),
                        'horsepower': car['specs'].get('horsepower'),
                        'torque': car['specs'].get('torque'),
                        'fuelConsumption': car['specs'].get('fuelConsumption'),
                        'dimensions': car['specs'].get('dimensions'),
                        'weight': car['specs'].get('weight')
                    }
                    writer.writerow(flat_row)
            print(f"   💾 Đã lưu nháp vào {filename}")
        except Exception as e:
            print(f"⚠️ Lỗi lưu file CSV: {e}")

if __name__ == "__main__":
    crawler = BonBanhAutoCrawler()
    crawler.run()