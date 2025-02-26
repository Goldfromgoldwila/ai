from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model without quantization
try:
    logger.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    logger.info("Loading model 'Salesforce/codet5-small'...")
    model = T5ForConditionalGeneration.from_pretrained(
        "Salesforce/codet5-small",
        torch_dtype=torch.float16,  # Half-precision to save memory
        device_map="auto"  # CPU-only
    )
    device = torch.device("cpu")
    model.to(device)
    logger.info(f"Model loaded on: {device}")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

@app.post("/rewrite")
async def rewrite_code(
    prompt: str = Form("Rewrite this Java code for Minecraft 1.20.1 compatibility: "),
    files: List[UploadFile] = None
):
    logger.info(f"Received request with prompt: '{prompt}', files: {[f.filename if f else None for f in files] if files else 'None'}")
    
    if not files:
        logger.warning("No files uploaded")
        return {"error": "No files uploaded"}
    if model is None:
        logger.error("Model is not loaded")
        return {"error": "Model failed to load, service unavailable"}

    results = []
    for file in files:
        try:
            content = await file.read()
            input_code = content.decode("utf-8")
            input_with_prompt = prompt + input_code
            logger.info(f"Processing file: {file.filename}, input length: {len(input_with_prompt)}")

            inputs = tokenizer(input_with_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            logger.info(f"Tokenized input for {file.filename}, tokens: {inputs['input_ids'].shape}")

            outputs = model.generate(
                inputs["input_ids"],
                max_length=512,
                num_beams=1,  # Reduce beams to minimize memory
                early_stopping=True
            )
            rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Processed {file.filename} successfully, output length: {len(rewritten)}")
            results.append({"filename": file.filename, "original": input_code, "rewritten": rewritten})
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "error": str(e)})

    logger.info(f"Returning results: {len(results)} files processed")
    return {"results": results}

@app.get("/")
async def root():
    status = "CodeT5+ server running" if model else "Server running, but model failed to load"
    logger.info(f"Root endpoint called, status: {status}")
    return {"message": status}