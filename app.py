from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from typing import List, Optional
import logging
import asyncio
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Log memory before loading
logger.info(f"Memory usage before model load: {psutil.virtual_memory().percent}%")

try:
    logger.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    logger.info("Tokenizer loaded. Loading model 'Salesforce/codet5-small'...")
    model = T5ForConditionalGeneration.from_pretrained(
        "Salesforce/codet5-small",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,  # Reduce memory spike
        device_map="auto"
    )
    device = torch.device("cpu")
    model.to(device)
    logger.info(f"Model loaded on: {device}. Memory usage: {psutil.virtual_memory().percent}%")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}. Memory usage: {psutil.virtual_memory().percent}%")
    model = None

@app.post("/rewrite")
async def rewrite_code(
    prompt: str = Form("Rewrite this Java code for Minecraft 1.20.1 compatibility: "),
    files: Optional[List[UploadFile]] = None
):
    logger.info(f"Received request with prompt: '{prompt}', files: {[f.filename if f else None for f in files] if files else 'None'}")
    
    if model is None:
        logger.error("Model is not loaded")
        response = {"error": "Model failed to load, service unavailable"}
        logger.info(f"Sending response: {response}")
        return response

    results = []
    # Same logic as before—unchanged for brevity
    if not files:
        try:
            prompt_lower = prompt.lower().strip()
            if "hello" in prompt_lower or "helo" in prompt_lower:
                rewritten = "Hello! I can analyze or rewrite Minecraft mod code—upload a Java file to get started!"
            elif "what can you do" in prompt_lower:
                rewritten = "I can analyze Java code for Minecraft mods, rewrite it for compatibility (e.g., 1.20.1), or suggest fixes. Try 'fix this code' with a file!"
            elif "fix" in prompt_lower or "code" in prompt_lower:
                rewritten = "Please upload a Java file for me to fix or analyze for Minecraft compatibility."
            elif "2+2" in prompt_lower:
                rewritten = "2+2 equals 4!"
            else:
                logger.info(f"Processing prompt alone, input length: {len(prompt)}")
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
                logger.info(f"Tokenized prompt, tokens: {inputs['input_ids'].shape}")
                max_length = min(50, max(10, len(prompt) * 2))
                async def generate_with_timeout():
                    return await asyncio.to_thread(model.generate, 
                        inputs["input_ids"],
                        max_length=max_length,
                        num_beams=1,
                        early_stopping=False
                    )
                outputs = await asyncio.wait_for(generate_with_timeout(), timeout=10.0)
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
        for file in files:
            try:
                content = await file.read()
                input_code = content.decode("utf-8")
                input_with_prompt = prompt + "\n" + input_code
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
                        early_stopping=False
                    )
                outputs = await asyncio.wait_for(generate_with_timeout(), timeout=60.0)
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