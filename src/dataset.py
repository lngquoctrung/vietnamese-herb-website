import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from google import genai
from google.genai import types
from .utils import get_logger, safe_path, validate_json_structure

EXTRACTION_PROMPT = """
Bạn là chuyên gia phân tích tài liệu Y học Cổ truyền Việt Nam.

VĂN BẢN: OCR từ PDF scan về thuốc Đông y.

CẤU TRÚC VỊ THUỐC:
- TIÊU ĐỀ: TÊN VỊ THUỐC (IN HOA) + chữ Hán
- Còn gọi là: tên khác
- Tên khoa học: Latin
- Thuộc họ: họ thực vật
- A. Mô tả cây
- B. Phân bố, thu hái, chế biến
- C. Thành phần hóa học
- D. Tác dụng dược lý
- E. Công dụng và liều dùng

QUY TẮC:
1. BUỘC PHẢI có đủ 5 phần A, B, C, D, E mới trích xuất
2. Văn bản BẮT ĐẦU giữa chừng (không có tiêu đề) -> BỎ QUA
3. Văn bản KẾT THÚC giữa chừng (thiếu phần E) -> BỎ QUA
4. Nếu một phần KHÔNG CÓ THÔNG TIN -> ghi rõ: "Không có thông tin"
5. KHÔNG được để trống hay null
6. Sửa lỗi chính tả OCR

JSON OUTPUT:
{{
  "vi_thuoc": [
    {{
      "ten_viet_nam": "TÊN IN HOA (không chữ Hán)",
      "ten_goi_khac": "Tên khác; phân cách bằng ;",
      "ten_khoa_hoc": "Tên Latin đầy đủ",
      "ho_thuc_vat": "Họ thực vật",
      "mo_ta": "Phần A - Mô tả chi tiết",
      "phan_bo_thu_hai_che_bien": "Phần B - Phân bố và thu hái",
      "thanh_phan_hoa_hoc": "Phần C - Thành phần hóa học",
      "tac_dung_duoc_ly": "Phần D - Tác dụng dược lý",
      "tinh_vi": "Tính vị từ phần E",
      "quy_kinh": "Kinh lạc từ phần E",
      "cong_dung": "Phần E - Công dụng điều trị",
      "lieu_dung": "Phần E - Liều dùng (gram/ngày)",
      "ghi_chu": "Chống chỉ định từ phần E"
    }}
  ],
  "bai_thuoc": [
    {{
      "ten_bai_thuoc": "Tên đơn thuốc",
      "chu_tri": "Bệnh trị",
      "nguon_goc": "Nguồn gốc",
      "ghi_chu": "Cách sắc, dùng"
    }}
  ],
  "cong_thuc": [
    {{
      "ten_bai_thuoc": "Tên đơn KHỚP bai_thuoc",
      "ten_vi_thuoc": "Tên vị KHỚP vi_thuoc",
      "lieu_luong": "Số lượng",
      "vai_tro": "Vai trò",
      "ghi_chu_che_bien": "Chế biến đặc biệt"
    }}
  ]
}}

CHÚ Ý:
- JSON hợp lệ, không thêm ```
- Tên NHẤT QUÁN giữa các bảng
- Nếu phần nào KHÔNG TỒN TẠI -> ghi "Không có thông tin"

VĂN BẢN:
***
{text}
***
"""

class DataExtractor:
    def __init__(self, config, logger_name=__name__, logger_path=None):
        self.config = config
        self.gemini_client = genai.Client(api_key=self.config.GEMINI_API_KEY)
        self.logger = get_logger(name=logger_name, filepath=logger_path)
        self.successful_chunks = 0
        self.failed_chunks = 0

    def _upload_and_wait(self, pdf_path: str, timeout: int = 300) -> Any:
        self.logger.info(f"Uploading PDF: {safe_path(pdf_path)}")
        uploaded_file = self.gemini_client.files.upload(file=pdf_path)
        self.logger.info(f"File uploaded successfully: {uploaded_file.name}")
        
        start_time = time.time()
        while hasattr(uploaded_file, 'state') and uploaded_file.state == 'PROCESSING':
            if time.time() - start_time > timeout:
                raise TimeoutError(f"File processing timeout after {timeout}s")
            time.sleep(2)
            uploaded_file = self.gemini_client.files.get(name=uploaded_file.name)
        
        return uploaded_file

    def _chunk_pdf_pages(self, uploaded_file: Any, pages_per_chunk: int = 6, overlap_pages: int = 2) -> List[Dict[str, Any]]:
        self.logger.info(f"Processing PDF with {pages_per_chunk} pages/chunk, {overlap_pages} pages overlap...")
        
        chunks = []
        total_pages = 200
        step = pages_per_chunk - overlap_pages
        
        start_page = 1
        chunk_position = 0
        
        while start_page <= total_pages:
            end_page = min(start_page + pages_per_chunk - 1, total_pages)
            
            extraction_prompt = f"""
                Trích xuất VĂN BẢN từ trang {start_page} đến trang {end_page} của PDF.

                YÊU CẦU:
                1. Trích xuất TOÀN BỘ nội dung từ các trang này
                2. GIỮ NGUYÊN:
                - Tiêu đề vị thuốc (IN HOA + chữ Hán)
                - Các phần A, B, C, D, E
                - Tên khoa học, công thức hóa học
                - Đơn thuốc và liều dùng
                3. LOẠI BỎ: Header, footer, số trang, watermark
                4. Nếu vị thuốc bị cắt giữa chừng -> giữ nguyên, đừng bỏ

                Bắt đầu trích xuất trang {start_page}-{end_page}:
            """
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.config.MODEL_NAME,
                        contents=[uploaded_file, extraction_prompt],
                        config=types.GenerateContentConfig(temperature=0.0)
                    )
                    
                    text = response.text.strip()
                    if len(text) > 100:
                        chunks.append({
                            'text': text,
                            'start_page': start_page,
                            'end_page': end_page,
                            'position': chunk_position
                        })
                        self.logger.info(f"Extracted pages {start_page}-{end_page}: {len(text)} chars")
                        chunk_position += 1
                    else:
                        self.logger.warning(f"Pages {start_page}-{end_page}: too short, skipping")
                    
                    time.sleep(10)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                        wait_time = 60 * (retry + 1)
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {retry+1}/{max_retries}")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Error extracting pages {start_page}-{end_page}: {e}")
                        break
            
            start_page += step
        
        self.logger.info(f"Created {len(chunks)} overlapping chunks")
        return chunks

    def _call_gemini_with_retry(self, prompt: str) -> Optional[Dict[str, Any]]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.logger.info(f"API call attempt {attempt + 1}/{self.config.MAX_RETRIES}")
                
                response = self.gemini_client.models.generate_content(
                    model=self.config.MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        max_output_tokens=self.config.MAX_OUTPUT_TOKENS
                    )
                )
                
                result = json.loads(response.text)
                result = validate_json_structure(result)
                
                self.logger.info(f"Success: {len(result['vi_thuoc'])} herbs extracted")
                time.sleep(self.config.DELAY_BETWEEN_REQUESTS)
                
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error: {e}")
                if hasattr(response, 'text'):
                    self.logger.debug(f"Response preview: {response.text[:200]}...")
                
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    wait_time = 60 * (attempt + 1)
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Error: {str(e)}")
                
            if attempt < self.config.MAX_RETRIES - 1:
                wait_time = (2 ** attempt) * self.config.INITIAL_BACKOFF
                self.logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                self.logger.error(f"Failed after {self.config.MAX_RETRIES} attempts")
        
        return None

    def _deduplicate_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Deduplicating results...")
        
        seen_herbs = set()
        unique_herbs = []
        for herb in data['vi_thuoc']:
            name = herb.get('ten_viet_nam', '').strip().lower()
            if name and name not in seen_herbs:
                seen_herbs.add(name)
                unique_herbs.append(herb)
            elif name in seen_herbs:
                self.logger.debug(f"Duplicate herb removed: {name}")
        
        seen_prescriptions = set()
        unique_prescriptions = []
        for prescription in data['bai_thuoc']:
            name = prescription.get('ten_bai_thuoc', '').strip().lower()
            if name and name not in seen_prescriptions:
                seen_prescriptions.add(name)
                unique_prescriptions.append(prescription)
        
        return {
            'vi_thuoc': unique_herbs,
            'bai_thuoc': unique_prescriptions,
            'cong_thuc': data['cong_thuc']
        }

    def _save_to_files(self, data: Dict[str, Any]):
        self.logger.info("Saving results")
        
        try:
            os.makedirs(os.path.dirname(self.config.OUTPUT_JSON), exist_ok=True)
            
            self.logger.info(f"Saving to {safe_path(self.config.OUTPUT_JSON)}...")
            with open(self.config.OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if data['vi_thuoc']:
                df = pd.DataFrame(data['vi_thuoc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(
                    self.config.OUTPUT_CSV_VI_THUOC,
                    index=False,
                    encoding='utf-8-sig',
                    quoting=1,
                    escapechar='\\'
                )
            
            if data['bai_thuoc']:
                df = pd.DataFrame(data['bai_thuoc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(
                    self.config.OUTPUT_CSV_BAI_THUOC,
                    index=False,
                    encoding='utf-8-sig',
                    quoting=1,
                    escapechar='\\'
                )
            
            if data['cong_thuc']:
                df = pd.DataFrame(data['cong_thuc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(
                    self.config.OUTPUT_CSV_CONG_THUC,
                    index=False,
                    encoding='utf-8-sig',
                    quoting=1,
                    escapechar='\\'
                )
                
        except Exception as e:
            self.logger.error(f"Save error: {e}")
            raise

    def process_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        self.logger.info(f"Processing: {safe_path(pdf_path)}")
        self.logger.info(f"Model: {self.config.MODEL_NAME}")
        self.logger.info(f"Rate limit: {self.config.REQUESTS_PER_MINUTE} RPM")
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            self.logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            uploaded_file = self._upload_and_wait(pdf_path)
            chunks = self._chunk_pdf_pages(uploaded_file, pages_per_chunk=6, overlap_pages=2)
            
            all_results = {
                'vi_thuoc': [],
                'bai_thuoc': [],
                'cong_thuc': []
            }
            
            self.successful_chunks = 0
            self.failed_chunks = 0
            
            for chunk_info in chunks:
                self.logger.info(f"Processing chunk {chunk_info['position'] + 1}/{len(chunks)}")
                self.logger.info(f"Pages {chunk_info['start_page']}-{chunk_info['end_page']}: {len(chunk_info['text'])} chars")
                
                prompt = EXTRACTION_PROMPT.format(text=chunk_info['text'])
                result = self._call_gemini_with_retry(prompt)
                
                if result:
                    all_results['vi_thuoc'].extend(result.get('vi_thuoc', []))
                    all_results['bai_thuoc'].extend(result.get('bai_thuoc', []))
                    all_results['cong_thuc'].extend(result.get('cong_thuc', []))
                    self.successful_chunks += 1
                else:
                    self.failed_chunks += 1
                    self.logger.warning("Chunk failed, skipping...")
            
            try:
                self.gemini_client.files.delete(name=uploaded_file.name)
                self.logger.info(f"Cleaned up: {uploaded_file.name}")
            except Exception as e:
                self.logger.warning(f"Cleanup warning: {e}")
            
            all_results = self._deduplicate_results(all_results)
            self._save_to_files(all_results)
            
            self.logger.info(f"Successful chunks: {self.successful_chunks}/{len(chunks)}")
            self.logger.info(f"Failed chunks: {self.failed_chunks}/{len(chunks)}")
            self.logger.info(f"Total herbs: {len(all_results['vi_thuoc'])}")
            self.logger.info(f"Total prescriptions: {len(all_results['bai_thuoc'])}")
            self.logger.info(f"Total formulas: {len(all_results['cong_thuc'])}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            raise
