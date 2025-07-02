## 0. Tạo và Kích Hoạt Môi Trường Ảo (Khuyến nghị)
Nên sử dụng môi trường ảo để quản lý dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # Trên macOS/Linux
# hoặc
.venv\Scripts\activate   # Trên Windows
```
## 1. Cài Đặt Dependencies
Trước khi chạy server, hãy đảm bảo bạn đã cài đặt tất cả các dependencies cần thiết:
```bash
pip install -r requirements.txt
```
## 2. Tải các file checkpoint của Restormer cho các tác vụ khôi phục ảnh
```bash
python download_model.py
```
## 3. Khởi Động Server
Chạy server trên cổng 8000 bằng lệnh:
```bash
python api.py
```
