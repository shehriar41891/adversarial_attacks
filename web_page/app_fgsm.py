from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from fgsm import custom_fgsm_attach
from functions import create_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify "http://localhost:8501" for streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = create_model()
loss_fn = torch.nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

class AttackResponse(BaseModel):
    success: bool
    original_label: int
    adversarial_label: int
    original_image: str
    adversarial_image: str
    confidence_original: float
    confidence_adversarial: float

def tensor_to_base64(tensor):
    from torchvision.transforms import ToPILImage
    import base64
    buffer = io.BytesIO()
    ToPILImage()(tensor.squeeze()).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

@app.post("/fgsm-attack/", response_model=AttackResponse)
async def perform_attack(
    image: UploadFile = File(...),
    label: int = Form(...),
    epsilon: float = Form(...)
):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    label_tensor = torch.tensor([label])

    adv_img = custom_fgsm_attach(model, loss_fn, img_tensor, label_tensor, epsilon)

    with torch.no_grad():
        output_orig = model(img_tensor)
        output_adv = model(adv_img)

    pred_orig = torch.argmax(output_orig, dim=1).item()
    pred_adv = torch.argmax(output_adv, dim=1).item()

    confidence_orig = torch.nn.functional.softmax(output_orig, dim=1)[0][pred_orig].item() * 100
    confidence_adv = torch.nn.functional.softmax(output_adv, dim=1)[0][pred_adv].item() * 100

    return AttackResponse(
        success=pred_orig != pred_adv,
        original_label=pred_orig,
        adversarial_label=pred_adv,
        original_image=tensor_to_base64(img_tensor),
        adversarial_image=tensor_to_base64(adv_img),
        confidence_original=confidence_orig,
        confidence_adversarial=confidence_adv
    )
