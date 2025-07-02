from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model import Restormer
from utils import get_default_args
import uvicorn
from pyngrok import ngrok, conf

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-PSNR", "X-SSIM", "X-Detected-Noise-Type"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CNN Model for Noise Detection ===
class NoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(pretrained=True)
        base.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base.last_channel, 1)
        )
        self.backbone = base

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# === Load CNN Models for Noise Detection ===
def load_cnn_models():
    models = {}
    cnn_model_paths = {
        "gaussian_denoise": "CNN_models/noise.pth",
        "derain": "CNN_models/rain.pth",
        "single_image_deblur": "CNN_models/blur.pth"
    }
    
    for task, path in cnn_model_paths.items():
        try:
            model = NoiseCNN().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models[task] = model
        except Exception as e:
            print(f"Warning: Could not load CNN model for {task}: {str(e)}")
    
    return models

# Load CNN models once at startup
cnn_models = load_cnn_models()

# === Transform for CNN models ===
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Load Restormer Model ===
def load_model(task: str):
    args = get_default_args()
    model = Restormer(
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        channels=args.channels,
        num_refinement=args.num_refinement,
        expansion_factor=args.expansion_factor
    )

    ckpt_paths = {
        "derain": "models/derain.pth",
        "gaussian_denoise": "models/gaussian_denoise.pth",
        "real_denoise": "models/real_denoise.pth",
        "motion_deblur": "models/motion_deblur.pth",
        "single_image_deblur": "models/single_image_deblur.pth"
    }
    ckpt_path = ckpt_paths.get(task)
    if not ckpt_path:
        raise ValueError("Unsupported task")

    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# === Image Processing Functions ===
def crop_to_multiple_of_8(img: Image.Image) -> Image.Image:
    w, h = img.size
    w = w - (w % 8)
    h = h - (h % 8)
    return img.crop((0, 0, w, h))

transform = transforms.Compose([
    transforms.ToTensor(),
])

def calculate_metrics(original_img: Image.Image, processed_img: Image.Image) -> dict:
    original_np = np.array(original_img)
    processed_np = np.array(processed_img)
    psnr_value = psnr(original_np, processed_np)
    ssim_value = ssim(original_np, processed_np, channel_axis=2)
    return {
        "psnr": float(psnr_value),
        "ssim": float(ssim_value)
    }

async def process_image_file(image_bytes: bytes, task: str):
    try:
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        cropped_image = crop_to_multiple_of_8(original_image)
        image_tensor = transform(cropped_image).unsqueeze(0).to(device)

        if image_tensor is None:
            raise HTTPException(status_code=500, detail="Failed to convert image to tensor")

        model = load_model(task)
        with torch.no_grad():
            output_tensor = model(image_tensor).squeeze().clamp(0, 1).cpu()

        if output_tensor is None:
            raise HTTPException(status_code=500, detail="Model failed to process image")

        output_image = transforms.ToPILImage()(output_tensor)
        metrics = calculate_metrics(cropped_image, output_image)
        
        buf = io.BytesIO()
        output_image.save(buf, format='PNG')
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "X-PSNR": str(metrics["psnr"]),
                "X-SSIM": str(metrics["ssim"])
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/auto-detect-and-process")
async def auto_detect_and_process(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_bytes = await file.read()
        print(f"\\n=== Processing Image ===")
        print(f"File: {file.filename}, Size: {len(image_bytes)} bytes")

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        image_tensor = cnn_transform(original_image).unsqueeze(0).to(device)
        
        print("\\n=== Scanning for Noise Types ===")
        detected_noise_type = None
        for noise_type, model in cnn_models.items():
            with torch.no_grad():
                output = model(image_tensor)
                probability = output.item()
                print(f"Checking {noise_type}: {probability:.4f}")
                
                if probability > 0.5:
                    detected_noise_type = noise_type
                    break
        
        if detected_noise_type is None:
            print("\\nNo noise detected in the image")
            return {
                "status": "no_noise_detected",
                "message": "No noise detected in the image"
            }
        
        print(f"\\n=== Detection Results ===")
        print(f"Detected noise type: {detected_noise_type}")
        
        # Define model pairs for each noise type
        model_pairs = {
            "gaussian_denoise": ["gaussian_denoise", "real_denoise"],
            "motion_deblur": ["motion_deblur", "single_image_deblur"],
            # "single_image_deblur": ["motion_deblur", "single_image_deblur"]
        }
        
        if detected_noise_type not in model_pairs:
            # For other noise types, process with single model
            result = await process_image_file(image_bytes, detected_noise_type)
            result.headers["X-Detected-Noise-Type"] = detected_noise_type
            return result
        
        # Process with both models
        best_result = None
        best_metrics = None
        best_model = None
        
        for model_name in model_pairs[detected_noise_type]:
            print(f"\\nProcessing with model: {model_name}")
            result = await process_image_file(image_bytes, model_name)
            psnr = float(result.headers["X-PSNR"])
            ssim = float(result.headers["X-SSIM"])
            
            if best_metrics is None or (psnr < best_metrics["psnr"] and 
                                      psnr < 40 ):
                best_result = result
                best_metrics = {"psnr": psnr, "ssim": ssim}
                best_model = model_name
        
        if best_metrics and best_metrics["psnr"] > 40 and b:
            return {
                "status": "no_noise_detected",
                "message": "No significant noise detected in the image"
            }
        
        print(f"\\n=== Final Processing Result ===")
        print(f"Used model: {best_model}")
        print(f"PSNR: {best_metrics['psnr']:.2f}")
        print(f"SSIM: {best_metrics['ssim']:.4f}")
        
        # Use the final model name as the official detected noise type
        best_result.headers["X-Detected-Noise-Type"] = best_model
        return best_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in auto detection and processing: {str(e)}")

# === Ngrok setup ===
def setup_ngrok():
    # Thay YOUR_AUTH_TOKEN bằng token của bạn từ https://dashboard.ngrok.com/get-started/your-authtoken
    conf.get_default().auth_token = "2p1nq5rK3ywq2n3HvcTGVDgtUGb_WPjQ2a39CcdETaTn4jFq"
    public_url = ngrok.connect(8000)
    print(f"\\n=== Ngrok Public URL ===")
    print(f"Public URL: {public_url}")
    return public_url

if __name__ == "__main__":
    # Setup ngrok
    public_url = setup_ngrok()
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
