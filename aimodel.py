from fastapi import FastAPI, UploadFile, Form
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from typing import List

app = FastAPI()

# Load quantized model at startup to squeeze into 512MB
try:
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
    model = T5ForConditionalGeneration.from_pretrained(
        "Salesforce/codet5p-220m",
        torch_dtype=torch.float16,  # Half-precision
        load_in_4bit=True,  # 4-bit quantization to reduce memory
        device_map="auto"  # Auto-select CPU (Render free has no GPU)
    )
    device = torch.device("cpu")  # Force CPU
    model.to(device)
    print("Model loaded on:", device)
except Exception as e:
    print(f"Model loading failed: {e}")

@app.post("/rewrite")
async def rewrite_code(
    prompt: str = Form("Rewrite this Java code for Minecraft 1.20.1 compatibility: "),
    files: List[UploadFile] = None
):
    if not files:
        return {"error": "No files uploaded"}

    results = []
    for file in files:
        content = await file.read()
        input_code = content.decode("utf-8")
        input_with_prompt = prompt + input_code

        try:
            inputs = tokenizer(input_with_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model.generate(
                inputs["input_ids"],
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"filename": file.filename, "original": input_code, "rewritten": rewritten})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}

@app.get("/")
async def root():
    return {"message": "CodeT5+ server running"}