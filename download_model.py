import gdown
import os

# Danh sách các model và link Google Drive (file ID)
model_links = {
    "models/gaussian_denoise.pth": "https://drive.google.com/uc?id=19cAaACVKJPjV11Y38mTasd7dFkKQUN98",
    "models/derain.pth": "https://drive.google.com/uc?id=1z5A-4JhA4eDle7y1vmUvcm4JaMfVCGsn",
    "models/real_denoise.pth": "https://drive.google.com/uc?id=11U4j_IeJfS-pUGBFklwDeFXhZTyIr70k",
    "models/motion_deblur.pth": "https://drive.google.com/uc?id=1FFhqI413ygzV6N_v6XVJ9NO9_wFy8a03",
    "models/single_image_deblur.pth": "https://drive.google.com/uc?id=1DC1_4ZfA69gE27dm1oi_ibZ3eOmYBc5E"
}

for path, url in model_links.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        gdown.download(url, path, quiet=False)
    else:
        print(f"{path} already exists.")
