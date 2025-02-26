from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from typing import List, Optional
import logging
import asyncio
import psutil  # For memory logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load model
try:
    logger.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    logger.info("Loading model 'Salesforce/codet5-small'...")
    model = T5ForConditionalGeneration.from_pretrained(
        "Salesforce/codet5-small",
        torch_dtype=torch.float16,
        device_map="auto"
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
    files: Optional[List[UploadFile]] = None  # Files are optional
):
    logger.info(f"Received request with prompt: '{prompt}', files: {[f.filename if f else None for f in files] if files else 'None'}")
    
    if model is None:
        logger.error("Model is not loaded")
        response = {"error": "Model failed to load, service unavailable"}
        logger.info(f"Sending response: {response}")
        return response

    results = []
    
    # Handle case with no files (process prompt alone)
    if not files:
        try:
            logger.info(f"Processing prompt alone, input length: {len(prompt)}")
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
            logger.info(f"Tokenized prompt, tokens: {inputs['input_ids'].shape}")

            async def generate_with_timeout():
                return await asyncio.to_thread(model.generate, 
                    inputs["input_ids"],
                    max_length=512,
                    num_beams=1,
                    early_stopping=True
                )
            
            outputs = await asyncio.wait_for(generate_with_timeout(), timeout=30.0)
            rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Processed prompt successfully, output length: {len(rewritten)}")
            results.append({"prompt": prompt, "rewritten": rewritten})
        except asyncio.TimeoutError:
            logger.error("Timeout processing prompt")
            results.append({"prompt": prompt, "error": "Processing timed out"})
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            results.append({"prompt": prompt, "error": str(e)})
    else:
        # Handle files if provided
        for file in files:
            try:
                content = await file.read()
                input_code = content.decode("utf-8")
                input_with_prompt = prompt + input_code
                logger.info(f"Processing file: {file.filename}, input length: {len(input_with_prompt)}")

                inputs = tokenizer(input_with_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
                logger.info(f"Tokenized input for {file.filename}, tokens: {inputs['input_ids'].shape}")
                memory_usage = psutil.virtual_memory().percent
                logger.info(f"Memory usage before generation: {memory_usage}%")

                async def generate_with_timeout():
                    return await asyncio.to_thread(model.generate, 
                        inputs["input_ids"],
                        max_length=512,
                        num_beams=1,
                        early_stopping=True
                    )
                
                outputs = await asyncio.wait_for(generate_with_timeout(), timeout=30.0)
                rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Processed {file.filename} successfully, output length: {len(rewritten)}")
                results.append({"filename": file.filename, "original": input_code, "rewritten": rewritten})
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing {file.filename}")
                results.append({"filename": file.filename, "error": "Processing timed out"})
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({"filename": file.filename, "error": str(e)})

    response = {"results": results}
    logger.info(f"Sending response: {response}")
    return response

@app.get("/")
async def root():
    status = "CodeT5+ server running" if model else "Server running, but model failed to load"
    logger.info(f"Root endpoint called, status: {status}")
    return {"message": status}

@app.options("/rewrite")
async def options_rewrite():
    logger.info("Received OPTIONS request for /rewrite")
    return {"message": "OPTIONS OK"}