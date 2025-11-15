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
Bạn là chuyên gia phân tích tài liệu Y học Cổ truyền Việt Nam với kinh nghiệm phân tích tài liệu scan.

THÔNG TIN VỀ VĂN BẢN:
- Đây là văn bản được OCR từ PDF scan (image-based)
- Cấu trúc: Sách về cây thuốc Đông y
- Mỗi vị thuốc có các phần: A. Mô tả cây, B. Phân bố thu hái, C. Thành phần hóa học, D. Tác dụng dược lý, E. Công dụng và liều dùng

NHIỆM VỤ: Trích xuất thông tin vị thuốc từ văn bản dưới đây.

QUY TẮC TRÍCH XUẤT:
1. CHỈ trích xuất vị thuốc HOÀN CHỈNH - có đủ các phần A, B, C, D, E
2. Nếu văn bản BẮT ĐẦU giữa chừng một vị thuốc (không có tiêu đề) -> BỎ QUA
3. Nếu văn bản KẾT THÚC giữa chừng một vị thuốc (thiếu phần E) -> BỎ QUA
4. Với OCR text, có thể có lỗi chính tả nhỏ -> sửa lại cho đúng nếu hiểu được
5. Tách rõ:
   - Tên Việt Nam (chữ IN HOA)
   - Tên Hán Việt (chữ Trung Quốc trong ngoặc hoặc liền sau)
   - Tên khoa học (chữ Latin, thường có L., Thunb., DC., v.v.)

CẤU TRÚC CỤ THỂ CỦA VỊ THUỐC:
- TIÊU ĐỀ: TÊN VỊ THUỐC (IN HOA) + chữ Hán (nếu có)
- Còn gọi là: các tên gọi khác
- Tên khoa học: tên Latin
- Thuộc họ: tên họ thực vật
- A. Mô tả cây: hình thái, đặc điểm nhận biết
- B. Phân bố, thu hái và chế biến: nơi mọc, thời gian thu, cách chế biến
- C. Thành phần hóa học: các hợp chất, công thức hóa học
- D. Tác dụng dược lý: kết quả nghiên cứu, thí nghiệm
- E. Công dụng và liều dùng: bệnh trị, liều lượng, đơn thuốc

ĐỊNH DẠNG JSON OUTPUT:
{
  "vi_thuoc": [
    {
      "ten_viet_nam": "Tên chính IN HOA (không có chữ Hán)",
      "ten_goi_khac": "Các tên khác, phân cách bởi dấu ;",
      "ten_khoa_hoc": "Tên Latin đầy đủ với tác giả",
      "ho_thuc_vat": "Tên họ thực vật (VD: Lamiaceae, Asteraceae)",
      "mo_ta": "Phần A - Mô tả hình thái chi tiết",
      "phan_bo_thu_hai_che_bien": "Phần B - Phân bố địa lý, thời gian thu hái, cách chế biến",
      "thanh_phan_hoa_hoc": "Phần C - Các hợp chất và công thức",
      "tac_dung_duoc_ly": "Phần D - Tác dụng dược lý đã nghiên cứu",
      "tinh_vi": "Tính (hàn/nhiệt/bình) và Vị (chua/đắng/ngọt/cay/mặn) - thường trong phần E",
      "quy_kinh": "Các kinh lạc: Tâm, Can, Tỳ, Phế, Thận - thường trong phần E",
      "cong_dung": "Phần E - Công dụng điều trị theo Đông y",
      "lieu_dung": "Phần E - Liều dùng cụ thể (gram/ngày)",
      "ghi_chu": "Chống chỉ định, cảnh báo từ phần E"
    }
  ],
  "bai_thuoc": [
    {
      "ten_bai_thuoc": "Tên đơn thuốc (thường trong phần E)",
      "chu_tri": "Bệnh điều trị",
      "nguon_goc": "Nguồn gốc đơn thuốc (nếu có)",
      "ghi_chu": "Cách sắc, cách dùng"
    }
  ],
  "cong_thuc": [
    {
      "ten_bai_thuoc": "Tên đơn thuốc (KHỚP với bảng bai_thuoc)",
      "ten_vi_thuoc": "Tên vị thuốc (KHỚP với bảng vi_thuoc)",
      "lieu_luong": "Số lượng cụ thể",
      "vai_tro": "Vai trò trong đơn (nếu có)",
      "ghi_chu_che_bien": "Chế biến đặc biệt (nếu có)"
    }
  ]
}

XỬ LÝ LỖI OCR:
- Sửa lỗi chính tả rõ ràng
- Giữ nguyên thuật ngữ chuyên môn
- Nếu không chắc -> giữ nguyên text gốc

CHÚ Ý:
- Không thêm ```
- Đảm bảo JSON hợp lệ
- Tên phải NHẤT QUÁN để liên kết được giữa các bảng

VĂN BẢN CẦN PHÂN TÍCH:
***
{text}
***
Hãy trích xuất cẩn thận theo cấu trúc A-B-C-D-E và trả về JSON.
"""

class DataExtractor:
    def __init__(self, config, logger_name=__name__, logger_path=None):
        self.config = config
        self.gemini_client = genai.Client(api_key=self.config.GEMINI_API_KEY)
        self.logger = get_logger(
            name=logger_name,
            filepath=logger_path
        )
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

    def _extract_text_with_structure(self, uploaded_file: Any) -> str:
        self.logger.info("Extracting text with structure preservation...")
        extraction_prompt = """
        Trích xuất TOÀN BỘ văn bản từ PDF này với yêu cầu:

        1. GIỮ NGUYÊN cấu trúc phân cấp:
           - Tiêu đề chính (VÍ DỤ: ÍCH MẪU 益母草)
           - Các phần A, B, C, D (Mô tả, Phân bố, Thành phần...)
           - Đơn thuốc và công thức
        2. ĐÁNH DẤU ranh giới:
           - Dùng === để ngăn cách giữa các vị thuốc
           - Dùng --- để ngăn cách các phần trong một vị thuốc
        3. GIỮ NGUYÊN:
           - Ký tự Hán Việt
           - Tên Latin (in nghiêng)
           - Số liệu, đơn vị đo
        4. LOẠI BỎ: Header, footer, số trang

        Format mẫu:
        ===
        TÊN VỊ THUỐC 汉字
        ---
        A. Mô tả...
        ---
        B. Phân bố...
        ===
        Hãy bắt đầu trích xuất.
        """
        response = self.gemini_client.models.generate_content(
            model=self.config.MODEL_NAME,
            contents=[uploaded_file, extraction_prompt],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        return response.text

    def _chunk_by_herbs(self, text: str) -> List[Dict[str, Any]]:
        self.logger.info("Chunking optimized for scanned PDF...")
        herb_pattern = r'(?:^|\n)([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ\\s]{3,50})(?:\\s+([一-鿿]{2,10}))?(?:\\s*\\n|\\s*$)'
        matches = list(re.finditer(herb_pattern, text, re.MULTILINE))
        
        chunks = []
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)

            herb_text = text[start_pos:end_pos].strip()
            herb_name = match.group(1).strip()
            has_structure = all([
                re.search(r'\\bA\\.\\s+(?:Mô tả|M t)', herb_text, re.IGNORECASE),
                re.search(r'\\bB\\.\\s+(?:Phân bố|Phn b)', herb_text, re.IGNORECASE),
                re.search(r'\\bC\\.\\s+(?:Thành phần|Thnh phn)', herb_text, re.IGNORECASE),
                re.search(r'\\bD\\.\\s+(?:Tác dụng|Tc dng)', herb_text, re.IGNORECASE),
                re.search(r'\\bE\\.\\s+(?:Công dụng|Cng dng)', herb_text, re.IGNORECASE)
            ])
            
            if has_structure and len(herb_text) > self.config.MIN_CHUNK_LENGTH:
                chunks.append({
                    'text': herb_text,
                    'herb_name': herb_name,
                    'position': i,
                    'has_complete_structure': True
                })
                self.logger.info(f"Found complete herb: {herb_name}")
            else:
                self.logger.warning(f"Incomplete herb structure, skipping: {herb_name}")
        
        if not chunks:
            self.logger.warning("No complete herbs found with A-B-C-D-E structure")
            return []
        
        self.logger.info(f"Created {len(chunks)} complete herb chunks")
        return chunks

    def _fixed_size_chunk_with_overlap(self, text: str) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        position = 0

        while start < len(text):
            end = start + self.config.MAX_CHUNK_SIZE
            chunk_text = text[start:end]
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'text': chunk_text,
                    'herb_name': f'Chunk_{position}',
                    'position': position
                })
                position += 1
            start = end - self.config.OVERLAP_SIZE

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
                self.logger.debug(f"Response preview: {response.text[:200]}...")

            except Exception as e:
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
                df.to_csv(self.config.OUTPUT_CSV_VI_THUOC, index=False, encoding='utf-8-sig')
            if data['bai_thuoc']:
                df = pd.DataFrame(data['bai_thuoc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(self.config.OUTPUT_CSV_BAI_THUOC, index=False, encoding='utf-8-sig')
            if data['cong_thuc']:
                df = pd.DataFrame(data['cong_thuc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(self.config.OUTPUT_CSV_CONG_THUC, index=False, encoding='utf-8-sig')
        except Exception as e:
            self.logger.error(f"Save error: {e}")
            raise

    def process_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        self.logger.info(f"Processing: {safe_path(pdf_path)}")
        self.logger.info(f"Model: {self.config.MODEL_NAME}")
        self.logger.info(f"Rate limit: {self.config.REQUESTS_PER_MINUTE} RPM")
        self.logger.info(f"Max chunk size: {self.config.MAX_CHUNK_SIZE} chars")
        self.logger.info(f"Overlap: {self.config.OVERLAP_SIZE} chars")

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            self.logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        try:
            uploaded_file = self._upload_and_wait(pdf_path)
            full_text = self._extract_text_with_structure(uploaded_file)
            self.logger.info(f"Extracted {len(full_text)} characters")
            chunks = self._chunk_by_herbs(full_text)
            all_results = {
                'vi_thuoc': [],
                'bai_thuoc': [],
                'cong_thuc': []
            }
            self.successful_chunks = 0
            self.failed_chunks = 0
            for chunk_info in chunks:
                self.logger.info(f"Chunk {chunk_info['position'] + 1}/{len(chunks)}: {chunk_info['herb_name']}")
                self.logger.info(f"Length: {len(chunk_info['text'])} chars")
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
