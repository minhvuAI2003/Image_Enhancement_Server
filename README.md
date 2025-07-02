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
Chạy server bằng lệnh:
```bash
python api.py
```
Sau khi chạy xong, địa chỉ của Server sẽ là https://localhost:8000
