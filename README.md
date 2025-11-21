# Cách chạy dự án

## Tải dự án

```bash
# Clone dự án
git clone https://github.com/lngquoctrung/vietnamese-herb-website.git

# Vào folder dự án
cd vietnamese-herb-website

# Mở VS Code
code .
```

## Tạo môi trường ảo và cài thư viện

Mở terminal trong VS Code

```bash
# Linux
python3 -m venv .venv

# Windows
python -m venv .venv

# Activate môi trường ảo
# Linux
source .venv/bin/active

# Windows
./.venv/Scritps/active

# Cài thư viện
pip install -r requirements.txt
```

## Chạy dự án

Gõ `Ctrl + Shift + P` mở cửa sổ **Command Palette**, search `Python: Select Interpreter`, gõ `Enter`, chọn cái nào có đường dẫn như phần **Activate môi trường ảo** phía trên (thường thì chọn cái nó có chữ **Recommended**). Vào notebook, phía bên trên góc bên phải chọn kernel tương tự như chọn Interpreter, nhấn `Run All` và đợi.
