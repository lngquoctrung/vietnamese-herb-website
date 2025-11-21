import os
import random
import json
import time
import re
import hashlib
import pandas as pd

from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Optional
from google.genai import types
from google import genai
from pypdf import PdfReader, PdfWriter
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
7. TUYỆT ĐỐI KHÔNG lặp lại một câu hoặc một đoạn văn quá 2 lần. Nếu văn bản gốc bị lỗi lặp, hãy chỉ lấy thông tin một lần duy nhất.

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
    def __init__(self, settings, logger_name: str = __name__, logger_path: str = None):
        self.settings = settings
        self.gemini_client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
        self.logger = get_logger(logger_name, logger_path)
        self.successful_chunks = 0
        self.failed_chunks = 0
        self.seen_hashes = set()

    def _split_pdf_file(self, input_path: str, output_dir: str, 
                        pages_per_chunk: int = 400, overlap_pages: int = 50) -> list:
        # Read pdf file
        reader = PdfReader(input_path)
        total_pages = len(reader.pages)

        # Create folder to store splitted pdf file
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        # Calculate step to have overlap
        step = pages_per_chunk - overlap_pages

        chunk_num = 0
        start_idx = 0
        while start_idx < total_pages:
            writer = PdfWriter()
            end_idx = min(start_idx + pages_per_chunk, total_pages)

            # Add pages to chunk
            for page_num in range(start_idx, end_idx):
                writer.add_page(reader.pages[page_num])

            # Filename and start-end pages information
            output_filename = f"chunk_{chunk_num:03d}_pages_{start_idx+1:04d}-{end_idx:04d}.pdf"
            output_path = os.path.join(output_dir, output_filename)

            # Write to pdf file
            with open(output_path, "wb") as output_file:
                writer.write(output_file)
            
            output_files.append(output_path)
            self.logger.info(f"Created: {output_filename} ({end_idx - start_idx} pages)")

            chunk_num += 1
            start_idx += step

            # If the last chunk is too small, merge with previous chunk
            if start_idx < total_pages and (total_pages - start_idx) < overlap_pages:
                self.logger.info(f"Merging last {total_pages - start_idx} pages into previous chunk")
                break
    
        self.logger.info(f"Total chunks: {len(output_files)}")
        self.logger.info(f"Total pages: {total_pages}")
        self.logger.info(f"Overlap: {overlap_pages} pages between chunks")
        
        return output_files

    def _upload_and_wait(self, pdf_path: str, timeout: int = 300) -> Any:
        self.logger.info(f"Uploading PDF: {safe_path(pdf_path)}")
        # Upload pdf file to caching of Gemini
        uploaded_file = self.gemini_client.files.upload(file=pdf_path)
        self.logger.info(f"File uploaded successfully: {uploaded_file.name}")
        
        # Waiting for file to be uploaded
        start_time = time.time()
        while hasattr(uploaded_file, 'state') and uploaded_file.state == 'PROCESSING':
            if time.time() - start_time > timeout:
                raise TimeoutError(f"File processing timeout after {timeout}s")
            time.sleep(2)
            uploaded_file = self.gemini_client.files.get(name=uploaded_file.name)
        
        return uploaded_file

    def _calculate_content_hash(self, content: str) -> str:
        normalized = content.lower().strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _is_duplicate(self, herb: Dict[str, Any], existing_herbs: List[Dict[str, Any]]) -> bool:
        name = herb.get('ten_viet_nam', '').strip()
        scientific_name = herb.get('ten_khoa_hoc', '').strip()
        
        if not name or not scientific_name:
            return True  # Invalid entry
        
        # Check exact match first
        for existing in existing_herbs:
            existing_name = existing.get('ten_viet_nam', '').strip()
            existing_sci = existing.get('ten_khoa_hoc', '').strip()
            
            if name.lower() == existing_name.lower():
                return True
            
            # Check scientific name match
            if scientific_name and existing_sci and scientific_name.lower() == existing_sci.lower():
                return True
            
            # Check fuzzy similarity if enabled
            if self.settings.USE_FUZZY_MATCHING:
                name_sim = self._calculate_similarity(name, existing_name)
                if name_sim >= self.settings.SIMILARITY_THRESHOLD:
                    self.logger.debug(f"Fuzzy duplicate: '{name}' ~ '{existing_name}' ({name_sim:.2f})")
                    return True
        
        return False
    
    def _merge_duplicate_herbs(self, herb1: Dict[str, Any], herb2: Dict[str, Any]) -> Dict[str, Any]:
        merged = herb1.copy()
        
        for key, value in herb2.items():
            # Skip if current value is placeholder
            if value and value != "Không có thông tin":
                existing = merged.get(key, "")
                # Replace if existing is empty or placeholder
                if not existing or existing == "Không có thông tin":
                    merged[key] = value
                # Merge if both have content and different
                elif existing != value and len(value) > len(existing):
                    merged[key] = value
        
        return merged

    def _chunk_pdf_pages(self, uploaded_file: Any, total_pages: int, 
                        process_pages_per_request: int = 8, 
                        process_overlap_pages: int = 3) -> List[Dict[str, Any]]:
        self.logger.info(f"Processing PDF: {total_pages} pages, "
                        f"{process_pages_per_request} pages/chunk, "
                        f"{process_overlap_pages} overlap")
        
        chunks = []
        step = process_pages_per_request - process_overlap_pages
        start_page = 1
        chunk_position = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while start_page <= total_pages:
            end_page = min(start_page + process_pages_per_request - 1, total_pages)
            
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
                5. TUYỆT ĐỐI KHÔNG lặp lại một câu hoặc một đoạn văn quá 2 lần. Nếu văn bản gốc bị lỗi lặp, hãy chỉ lấy thông tin một lần duy nhất.

                Bắt đầu trích xuất trang {start_page}-{end_page}:
            """
            
            for retry in range(self.settings.MAX_RETRIES):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.settings.OCR_MODEL_NAME,
                        contents=[uploaded_file, extraction_prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=self.settings.MAX_OUTPUT_TOKENS
                        )
                    )
                    
                    text = response.text.strip()
                    
                    if len(text) > 100:
                        chunks.append({
                            'text': text,
                            'start_page': start_page,
                            'end_page': end_page,
                            'position': chunk_position,
                            'hash': self._calculate_content_hash(text)
                        })
                        self.logger.info(f"Pages {start_page}-{end_page}: {len(text)} chars")
                        chunk_position += 1
                        consecutive_failures = 0
                    else:
                        self.logger.warning(f"Pages {start_page}-{end_page}: too short ({len(text)} chars)")
                    
                    # Dynamic delay based on success
                    time.sleep(self.settings.DELAY_BETWEEN_REQUESTS)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    
                    if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                        # Exponential backoff with jitter
                        wait_time = min(
                            self.settings.INITIAL_BACKOFF * (self.settings.BACKOFF_MULTIPLIER ** retry),
                            self.settings.MAX_BACKOFF
                        )
                        jitter = wait_time * 0.1 * (0.5 - random.random())
                        total_wait = wait_time + jitter
                        
                        self.logger.warning(f"Rate limit hit, waiting {total_wait:.1f}s "
                                          f"(retry {retry+1}/{self.settings.MAX_RETRIES})")
                        time.sleep(total_wait)
                    else:
                        self.logger.error(f"Error pages {start_page}-{end_page}: {e}")
                        consecutive_failures += 1
                        
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error(f"Too many consecutive failures, stopping chunk extraction")
                            return chunks
                        break
            
            start_page += step
        
        self.logger.info(f"Created {len(chunks)} chunks from {total_pages} pages")
        return chunks

    def _call_gemini_with_retry(self, prompt: str, chunk_info: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        last_exception = None

        for attempt in range(self.settings.MAX_RETRIES):
            try:
                self.logger.info(f"API call {attempt + 1}/{self.settings.MAX_RETRIES}")
                current_temperature = 0.0 if attempt == 0 else min(0.1 + (attempt * 0.15), 1.0)

                response = self.gemini_client.models.generate_content(
                    model=self.settings.TEXT_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=current_temperature,
                        response_mime_type="application/json",
                        max_output_tokens=self.settings.MAX_OUTPUT_TOKENS,
                    )
                )
                # Parse and validate
                result = json.loads(response.text)
                result = validate_json_structure(result)
                
                # Log success
                herbs_count = len(result.get('vi_thuoc', []))
                prescriptions_count = len(result.get('bai_thuoc', []))
                formulas_count = len(result.get('cong_thuc', []))
                
                self.logger.info(f"Extracted: {herbs_count} herbs, "
                               f"{prescriptions_count} prescriptions, "
                               f"{formulas_count} formulas")
                
                # Add metadata if provided
                if chunk_info:
                    result['_metadata'] = {
                        'start_page': chunk_info.get('start_page'),
                        'end_page': chunk_info.get('end_page'),
                        'position': chunk_info.get('position')
                    }
                
                time.sleep(self.settings.DELAY_BETWEEN_REQUESTS)
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error: {e}")
                if hasattr(response, 'text'):
                    self.logger.debug(f"Response preview: {response.text[:300]}...")
                last_exception = e
                
            except Exception as e:
                error_str = str(e)
                last_exception = e
                
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    wait_time = min(
                        self.settings.INITIAL_BACKOFF * (self.settings.BACKOFF_MULTIPLIER ** attempt),
                        self.settings.MAX_BACKOFF
                    )
                    jitter = wait_time * 0.1 * (0.5 - random.random())
                    total_wait = wait_time + jitter
                    
                    self.logger.warning(f"Rate limit, waiting {total_wait:.1f}s...")
                    time.sleep(total_wait)
                else:
                    self.logger.error(f"API error: {str(e)}")
                    if attempt < self.settings.MAX_RETRIES - 1:
                        wait_time = self.settings.INITIAL_BACKOFF * (2 ** attempt)
                        self.logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
        
        self.logger.error(f"Failed after {self.settings.MAX_RETRIES} attempts")
        return None

    def _deduplicate_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Deduplicating results with similarity matching...")
        
        unique_herbs = []
        duplicate_count = 0
        
        for herb in data['vi_thuoc']:
            if not self._is_duplicate(herb, unique_herbs):
                unique_herbs.append(herb)
            else:
                duplicate_count += 1
                # Try to merge information
                for i, existing in enumerate(unique_herbs):
                    if self._calculate_similarity(
                        herb.get('ten_viet_nam', ''),
                        existing.get('ten_viet_nam', '')
                    ) >= self.settings.SIMILARITY_THRESHOLD:
                        unique_herbs[i] = self._merge_duplicate_herbs(existing, herb)
                        self.logger.debug(f"Merged duplicate: {herb.get('ten_viet_nam')}")
                        break
        
        self.logger.info(f"Removed {duplicate_count} duplicates, kept {len(unique_herbs)} unique herbs")
        
        # Deduplicate prescriptions
        seen_prescriptions = {}
        unique_prescriptions = []
        
        for prescription in data['bai_thuoc']:
            name = prescription.get('ten_bai_thuoc', '').strip().lower()
            if name and name not in seen_prescriptions:
                seen_prescriptions[name] = True
                unique_prescriptions.append(prescription)
        
        # Deduplicate formulas based on combination of prescription + herb + amount
        seen_formulas = set()
        unique_formulas = []
        
        for formula in data['cong_thuc']:
            key = (
                formula.get('ten_bai_thuoc', '').strip().lower(),
                formula.get('ten_vi_thuoc', '').strip().lower(),
                formula.get('lieu_luong', '').strip()
            )
            if key not in seen_formulas and key[0] and key[1]:
                seen_formulas.add(key)
                unique_formulas.append(formula)
        
        return {
            'vi_thuoc': unique_herbs,
            'bai_thuoc': unique_prescriptions,
            'cong_thuc': unique_formulas
        }

    def _save_to_files(self, data: Dict[str, Any]):
        self.logger.info("Saving results")
        
        try:
            os.makedirs(os.path.dirname(self.settings.OUTPUT_JSON), exist_ok=True)
            
            self.logger.info(f"Saving to {safe_path(self.settings.OUTPUT_JSON)}...")
            with open(self.settings.OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if data['vi_thuoc']:
                df = pd.DataFrame(data['vi_thuoc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(
                    self.settings.OUTPUT_CSV_VI_THUOC,
                    index=False,
                    encoding='utf-8-sig',
                    quoting=1,
                    escapechar='\\'
                )
            
            if data['bai_thuoc']:
                df = pd.DataFrame(data['bai_thuoc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(
                    self.settings.OUTPUT_CSV_BAI_THUOC,
                    index=False,
                    encoding='utf-8-sig',
                    quoting=1,
                    escapechar='\\'
                )
            
            if data['cong_thuc']:
                df = pd.DataFrame(data['cong_thuc'])
                df.insert(0, 'id', range(1, len(df) + 1))
                df.to_csv(
                    self.settings.OUTPUT_CSV_CONG_THUC,
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
        self.logger.info(f"Model for OCR: {self.settings.OCR_MODEL_NAME}")
        self.logger.info(f"Model for text extraction: {self.settings.TEXT_MODEL_NAME}")
        self.logger.info(f"Rate limit: {self.settings.REQUESTS_PER_MINUTE} RPM")
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Split large PDF into chunks
        splitted_pdf_paths = self._split_pdf_file(
            input_path=pdf_path_obj,
            output_dir=f"{self.settings.RAW_DATA_PATH}/chunks",
            pages_per_chunk=self.settings.PAGES_PER_CHUNK,
            overlap_pages=self.settings.OVERLAP_PAGES
        )
        
        # Initialize results OUTSIDE the loop
        all_results = {
            'vi_thuoc': [],
            'bai_thuoc': [],
            'cong_thuc': []
        }
        
        total_successful_chunks = 0
        total_failed_chunks = 0
        
        # Process each PDF chunk
        for chunk_idx, pdf_chunk_path in enumerate(splitted_pdf_paths):
            self.logger.info(f"Processing chunk {chunk_idx + 1}/{len(splitted_pdf_paths)}: {safe_path(pdf_chunk_path)}")
            
            try:
                pdf_reader = PdfReader(pdf_chunk_path)
                total_pages = len(pdf_reader.pages)
                
                # Upload to Gemini
                uploaded_file = self._upload_and_wait(pdf_chunk_path)
                
                # Extract text chunks with overlap
                text_chunks = self._chunk_pdf_pages(
                    uploaded_file=uploaded_file,
                    total_pages=total_pages,
                    process_pages_per_request=self.settings.PROCESS_PAGES_PER_REQUEST,
                    process_overlap_pages=self.settings.PROCESS_OVERLAP_PAGES,
                )
                
                # Process each text chunk
                chunk_successful = 0
                chunk_failed = 0
                
                for chunk_info in text_chunks:
                    self.logger.info(f"Processing text chunk {chunk_info['position'] + 1}/{len(text_chunks)}")
                    self.logger.info(f"Pages {chunk_info['start_page']}-{chunk_info['end_page']}: "
                                   f"{len(chunk_info['text'])} chars")
                    
                    prompt = EXTRACTION_PROMPT.format(text=chunk_info['text'])
                    result = self._call_gemini_with_retry(prompt, chunk_info)
                    
                    if result:
                        all_results['vi_thuoc'].extend(result.get('vi_thuoc', []))
                        all_results['bai_thuoc'].extend(result.get('bai_thuoc', []))
                        all_results['cong_thuc'].extend(result.get('cong_thuc', []))
                        chunk_successful += 1
                    else:
                        chunk_failed += 1
                        self.logger.warning("Text chunk failed, continuing...")
                
                total_successful_chunks += chunk_successful
                total_failed_chunks += chunk_failed
                
                self.logger.info(f"Chunk summary: {chunk_successful} successful, {chunk_failed} failed")
                
                # Cleanup uploaded file
                try:
                    self.gemini_client.files.delete(name=uploaded_file.name)
                    self.logger.info(f"Cleaned up: {uploaded_file.name}")
                except Exception as e:
                    self.logger.warning(f"Cleanup warning: {e}")
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                total_failed_chunks += 1
                continue
        
        # Deduplicate and save
        self.logger.info("Post-processing results...")
        
        self.logger.info(f"Before deduplication: "
                        f"{len(all_results['vi_thuoc'])} herbs, "
                        f"{len(all_results['bai_thuoc'])} prescriptions, "
                        f"{len(all_results['cong_thuc'])} formulas")
        
        all_results = self._deduplicate_results(all_results)
        self._save_to_files(all_results)
        
        # Summary
        self.logger.info(f"Successful chunks: {total_successful_chunks}")
        self.logger.info(f"Failed chunks: {total_failed_chunks}")
        self.logger.info(f"Total herbs: {len(all_results['vi_thuoc'])}")
        self.logger.info(f"Total prescriptions: {len(all_results['bai_thuoc'])}")
        self.logger.info(f"Total formulas: {len(all_results['cong_thuc'])}")
        
        return all_results