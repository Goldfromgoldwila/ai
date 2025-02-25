from fastapi import FastAPI, UploadFile, Form
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from typing import List

app = FastAPI()

# Load smaller, quantized model
try:
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    model = T5ForConditionalGeneration.from_pretrained(
        "Salesforce/codet5-small",
        torch_dtype=torch.float16,  # Half-precision
        load_in_8bit=True,  # 8-bit quantization
        device_map="auto"  # CPU-only
    )
    device = torch.device("cpu")
    model.to(device)
    print("Model loaded on:", device)
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

@app.post("/rewrite")
async def rewrite_code(
    prompt: str = Form("Rewrite this Java code for Minecraft 1.20.1 compatibility: "),
    files: List[UploadFile] = None
):
    if not files:
        return {"error": "No files uploaded"}
    if model is None:
        return {"error": "Model failed to load, service unavailable"}

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
                num_beams=2,  # Lower beams to save memory
                early_stopping=True
            )
            rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({"filename": file.filename, "original": input_code, "rewritten": rewritten})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"results": results}

@app.get("/")
async def root():
    return {"message": "CodeT5+ server running" if model else "Server running, but model failed to load"}