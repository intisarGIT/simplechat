# fantasy_chatbot_enhanced.py - Modernized UI with two-tab layout and enhanced character creation
import os
import shutil
import threading
import time
import uuid
import re
import base64
import glob
from typing import Optional, List, Dict, Any
from functools import lru_cache

import cv2
import numpy as np
import requests
import gradio as gr
try:
    import torch
except Exception:
    torch = None
try:
    from gradio_client import Client, handle_file
except Exception:
    Client = None
    handle_file = None
    print("Warning: gradio_client not available - Hugging Face fallback disabled")
    print("Optional dependency 'torch' not available — continuing without it.")

# Optional hard-coded RapidAPI key (NOT recommended). Leave empty and prefer env/config vars.
RAPIDAPI_HARDCODE = ""  # If you really want to hardcode a key, place it here (not recommended)
try:
    # Load .env automatically if python-dotenv is installed and a .env file exists
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except Exception:
    # If dotenv isn't installed or fails, continue — env vars may be set in the environment
    pass

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from io import BytesIO
from PIL import Image

# Local Stable Diffusion support has been removed. Image generation uses ImageRouter
# (remote) and Face++ for face merging. The diffusers/xformers imports and
# related initialization have been intentionally removed to avoid heavy local
# dependencies.

# Set up directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
MODELS_DIR = "models"
for directory in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI
app = FastAPI()
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
# Initialize face models variables but don't load them yet
# local face model variables removed (insightface pruning)


# Global settings and state
class AppState:
    def __init__(self):
        self.face_image_path = None
        self.physical_description = ""
        self.behavioral_description = ""
        self.character_name = ""
        self.relation_to_user = ""
        self.user_name = ""
        self.chat_context = ""
        self.chat_history = []
        self.image_history = []
        self.sd_model_path = ""
        # Fixed generation settings as requested
        self.scheduler_type = "dpm_2m_karras"
        self.guidance_scale = 7.0
        self.num_inference_steps = 14
        self.clip_skip = 2
        # Face swap components removed (we use Face++ remote merging)
        self.face_app = None
        self.face_swapper = None
        self.source_face = None
        self.source_img = None
        self.sd_pipe = None
        self.prompt_cache = {}  # Cache for prompt->image mapping
        self.last_used_prompt = None
        # New character attributes
        self.initial_attire = ""
        self.gender = "Female"  # Default gender
        self.style = "Photorealistic"  # Default style
        self.character_base_prompt = ""
        self.character_seed = None
        self.available_models = []  # Will store available SD models
        # ImageRouter API key (can be set via /set_api_settings or env IMAGEROUTER_API_KEY)
        # For security, prefer setting these as environment variables on the server
        # rather than keeping keys in source control.
        self.imagerouter_api_key = os.getenv("IMAGEROUTER_API_KEY", "")
        # Face++ credentials (use FACEPP_API_KEY and FACEPP_API_SECRET env vars)
        self.facepp_api_key = os.getenv("FACEPP_API_KEY", "")
        self.facepp_api_secret = os.getenv("FACEPP_API_SECRET", "")
        # RapidAPI (FaceSwap) credentials - primary faceswap service if provided
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY", "")
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "faceswap-image-transformation-api-free-api-face-swap.p.rapidapi.com")
        # Mistral API Key (use MISTRAL_API_KEY env var)
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
        # Hugging Face token for private spaces (use HUGGINGFACE_TOKEN env var)
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", "")
    



    # initialize_face_models removed — no local face model initialization needed

    def initialize_sd_model(self):
        """No-op: local Stable Diffusion initialization removed.

        This function exists for backward compatibility but will not attempt
        to load any local Stable Diffusion models. Image generation uses
        the remote ImageRouter API instead.
        """
        print("initialize_sd_model called but local SD support has been removed.")
        return False

    def scan_available_models(self):
        """Return an empty list - local SD models are not used."""
        self.available_models = []
        return self.available_models

# Create an instance of AppState (only once)
app_state = AppState()



def message_requests_image(msg: str) -> bool:
    """Return True if the user's message contains clear visual/image requests."""
    m = msg.lower()
    triggers = [
        "show", "show me", "picture", "look like", "appearance", "see me", "imagine", "visualize",
        "what i look like", "strip", "undress", "take off", "remove clothes", "no panties", "bottomless",
        "naked", "nude", "expose"
    ]
    return any(t in m for t in triggers)

# Change from a method to a standalone function that uses app_state
def generate_character_preview():
    """Generate a face-swapped character preview image based on inputs"""  
    try:
        # No local face models required (Face++ remote merging is used)
                
        # Check if SD model is initialized
        if not app_state.sd_pipe and app_state.sd_model_path:
            if not app_state.initialize_sd_model():
                return None, "Failed to initialize Stable Diffusion model"
        
        # Check if face image path exists
        if not app_state.face_image_path:
            return None, "Please upload a character face image first."
            
        if not os.path.exists(app_state.face_image_path):
            return None, f"Face image not found at {app_state.face_image_path}"

        # Load the uploaded image (no local face detection). extract_face_from_image
        # now returns (None, img_rgb) for compatibility.
        print(f"Loading face image from: {app_state.face_image_path}")
        extracted_face, source_img = extract_face_from_image(app_state.face_image_path)
        if source_img is None:
            return None, "Could not read the uploaded image."

        # We no longer have a local face object; store the image for later use.
        app_state.source_face = None
        app_state.source_img = source_img

        # Construct preview prompt
        preview_prompt = f"{app_state.gender}, standing, {app_state.physical_description}, wearing {app_state.initial_attire}, {app_state.style}, high quality, intricate details"
        app_state.character_base_prompt = preview_prompt

        # Always generate a new random seed each time this function is called
        import random
        import time
        # Use time as part of the seed to ensure uniqueness
        seed = random.randint(100000, 999999) + int(time.time()) % 10000
        app_state.character_seed = seed
        print(f"Using new random seed for character preview: {seed}")

        # Force regeneration by removing any cached versions of this prompt
        # Check for both the prompt alone and prompt_seed combinations
        for key in list(app_state.prompt_cache.keys()):
            if key == preview_prompt or key.startswith(f"{preview_prompt}_"):
                print(f"Removing cached prompt key '{key}' to force regeneration with new seed")
                app_state.prompt_cache.pop(key, None)

        # Generate image with face swap using the preview prompt
        image_path, msg = generate_image_with_face_swap(preview_prompt, seed=seed)
        if image_path:
            return os.path.join(OUTPUT_DIR, image_path), msg
        return None, msg
    except Exception as e:
        print("Error during character preview generation:", str(e))
        import traceback
        print(traceback.format_exc())
        return None, str(e)


def generate_preview_from_ui(face_upload_value=None, physical_description=None, initial_attire=None, gender_value=None, style_value=None):
    """UI handler for character preview generation that ensures models are loaded"""
    try:
        # Debug the inputs
        print(f"- face_upload_value: {face_upload_value}")
        print(f"- physical_description: {physical_description}")
        print(f"- initial_attire: {initial_attire}")
        print(f"- gender_value: {gender_value}")
        print(f"- style_value: {style_value}")
        
        # Make sure we have a Stable Diffusion model loaded
        if not app_state.sd_pipe:
            if not app_state.sd_model_path:
                return None, "Please select a Stable Diffusion model first"
            
            # Initialize the SD model
            if not app_state.initialize_sd_model():
                return None, "Failed to initialize Stable Diffusion model"
        
        # No local face models required here
        
        # Update app_state with the face image path if provided
        if face_upload_value is not None:
            # Ensure we have a valid file path
            if face_upload_value and os.path.exists(face_upload_value):
                app_state.face_image_path = face_upload_value
                print(f"Updated face_image_path to: {app_state.face_image_path}")
            else:
                print(f"Warning: Invalid face image path: {face_upload_value}")
                return None, "Please upload a valid character face image first."
        else:
            print("No face upload value provided")
            return None, "Please upload a character face image first."
        
        # Update other character attributes if provided
        if physical_description is not None:
            print(f"[DEBUG] Setting physical_description in generate_character_preview: '{physical_description}'")
            app_state.physical_description = physical_description
        if initial_attire is not None:
            app_state.initial_attire = initial_attire
        if gender_value is not None:
            app_state.gender = gender_value
        if style_value is not None:
            app_state.style = style_value
            
        # Now generate the preview
        return generate_character_preview()
    except Exception as e:
        print(f"Error in generate_preview_from_ui: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, f"Error generating preview: {str(e)}"


def extract_face_from_image(image_path):
    """Lightweight image reader: return (None, img_rgb).

    The original implementation used insightface to detect and return a face object
    and embeddings. That dependency has been removed — callers should now rely on
    app_state.face_image_path (the uploaded image file) rather than an insightface
    face object. For compatibility we return (None, img_rgb) so existing code that
    expects a source_img still works.
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None, img_rgb
    except Exception as e:
        print(f"Error reading image in extract_face_from_image: {e}")
        return None, None


def huggingface_face_swap(source_path, target_path, output_path):
    """
    Fallback face swap using Hugging Face Spaces API
    """
    try:
        if Client is None:
            print("Hugging Face gradio_client not available")
            return False
            
        print("Attempting Hugging Face face swap fallback...")
        
        # Get authentication token for private spaces
        hf_token = app_state.huggingface_token or os.getenv("HUGGINGFACE_TOKEN", "")
        
        # Initialize client with authentication if token is provided
        if hf_token:
            print("Using authenticated Hugging Face client (private space)")
            client = Client("intisarhasnain/face-swap2", hf_token=hf_token)
        else:
            print("Using public Hugging Face client (no authentication)")
            client = Client("intisarhasnain/face-swap2")
        
        # Call the prediction API
        result = client.predict(
            sourceImage=handle_file(source_path),
            sourceFaceIndex=1,
            destinationImage=handle_file(target_path),
            destinationFaceIndex=1,
            api_name="/predict"
        )
        
        print(f"Hugging Face API result: {result}")
        
        # Result should be a dict with 'path' key
        if isinstance(result, dict) and 'path' in result:
            result_path = result['path']
            if os.path.exists(result_path):
                # Copy the result to our output path
                shutil.copy(result_path, output_path)
                print(f"Hugging Face face swap successful, saved to {output_path}")
                return True
            else:
                print(f"Hugging Face result path does not exist: {result_path}")
                return False
        else:
            print(f"Unexpected Hugging Face result format: {result}")
            return False
            
    except Exception as e:
        print(f"Hugging Face face swap failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def swap_face(source_face, source_img, target_path, output_path):
    """Swap face using RapidAPI, fallback to Hugging Face Spaces API on failure."""
    try:
        # We no longer need Face++ credentials - remove this check
        # The function now uses RapidAPI as primary and Hugging Face as fallback

        # Ensure target image exists
        if not os.path.exists(target_path):
            alternative_path = os.path.join(OUTPUT_DIR, os.path.basename(target_path))
            if os.path.exists(alternative_path):
                target_path = alternative_path
            else:
                raise FileNotFoundError(f"Target image not found at {target_path}")

        # Create temporary files for template (source face crop) and merge image (target)
        tmp_dir = os.path.join(UPLOAD_DIR, "tmp_facepp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Prefer using the original uploaded face image as the template for Face++
        if getattr(app_state, 'face_image_path', None) and os.path.exists(app_state.face_image_path):
            template_path = app_state.face_image_path
        else:
            # Save source face crop to temp file
            template_path = os.path.join(tmp_dir, f"template_{uuid.uuid4()}.jpg")
            try:
                # source_img is RGB numpy array (from extract_face_from_image)
                if isinstance(source_img, np.ndarray):
                    src_crop = source_img.copy()
                    Image.fromarray(src_crop).save(template_path, format="JPEG")
                else:
                    # fallback if source_img is a path
                    shutil.copy(source_img, template_path)
            except Exception:
                # If we can't write the full source image, try to crop from bounding box
                try:
                    bbox = source_face.bbox.astype(int)
                    crop = source_img[max(0, bbox[1]):min(source_img.shape[0], bbox[3]), max(0, bbox[0]):min(source_img.shape[1], bbox[2])]
                    Image.fromarray(crop).save(template_path, format="JPEG")
                except Exception as e:
                    raise RuntimeError(f"Failed to prepare template face for Face++: {e}")

        # Ensure we have the full target image file
        merge_image_path = os.path.join(tmp_dir, f"merge_{uuid.uuid4()}.jpg")
        shutil.copy(target_path, merge_image_path)

        # --- Try RapidAPI FaceSwap first (if configured) ---
        try:
            rapidapi_key = RAPIDAPI_HARDCODE or app_state.rapidapi_key or os.getenv("RAPIDAPI_KEY", "")
            rapidapi_host = app_state.rapidapi_host or os.getenv("RAPIDAPI_HOST", "faceswap-image-transformation-api1.p.rapidapi.com")
            print(f"RapidAPI key present: {bool(rapidapi_key)}; host={rapidapi_host}")
            if rapidapi_key:
                # Prefer base64 endpoint since we have local files; build payload from files
                with open(template_path, "rb") as f:
                    src_b64 = base64.b64encode(f.read()).decode('utf-8')
                with open(merge_image_path, "rb") as f:
                    tgt_b64 = base64.b64encode(f.read()).decode('utf-8')

                # First, try the alternate RapidAPI provider (multipart file upload) as PRIMARY
                try:
                    alt_host_primary = "faceswap-image-transformation-api-free-api-face-swap.p.rapidapi.com"
                    alt_url_primary = f"https://{alt_host_primary}/api/face-swap/create"
                    headers_alt = {
                        "x-rapidapi-key": rapidapi_key,
                        "x-rapidapi-host": alt_host_primary
                    }
                    print(f"Attempting RapidAPI (free) multipart faceswap at {alt_url_primary}")
                    # Build raw multipart body with explicit boundary to match provider expectations
                    boundary = '---011000010111000001101001'
                    try:
                        with open(template_path, 'rb') as src_f, open(merge_image_path, 'rb') as tgt_f:
                            src_bytes = src_f.read()
                            tgt_bytes = tgt_f.read()

                        parts = []
                        parts.append(f"--{boundary}\r\n".encode('utf-8'))
                        parts.append(b'Content-Disposition: form-data; name="source_image"; filename="source.jpg"\r\n')
                        parts.append(b'Content-Type: image/jpeg\r\n\r\n')
                        parts.append(src_bytes)
                        parts.append(b"\r\n")
                        parts.append(f"--{boundary}\r\n".encode('utf-8'))
                        parts.append(b'Content-Disposition: form-data; name="target_image"; filename="target.jpg"\r\n')
                        parts.append(b'Content-Type: image/jpeg\r\n\r\n')
                        parts.append(tgt_bytes)
                        parts.append(b"\r\n")
                        parts.append(f"--{boundary}--\r\n".encode('utf-8'))
                        body = b''.join(parts)

                        headers_alt['Content-Type'] = f'multipart/form-data; boundary={boundary}'
                        resp_alt = requests.post(alt_url_primary, headers=headers_alt, data=body, timeout=120)
                    except Exception as rexc:
                        print(f"Error preparing or sending raw multipart to RapidAPI (free): {rexc}")
                        raise

                    print(f"RapidAPI (free) primary response status: {resp_alt.status_code}")
                    try:
                        j_alt = resp_alt.json()
                    except Exception:
                        j_alt = None

                    if j_alt:
                        print(f"RapidAPI (free) primary response keys: {list(j_alt.keys())}")
                        code = j_alt.get('code')
                        data_block = j_alt.get('data') or {}
                        image_url = None
                        if isinstance(data_block, dict):
                            image_url = data_block.get('image_url') or data_block.get('url')

                        if isinstance(image_url, str) and image_url.startswith('http'):
                            try:
                                dl_alt = requests.get(image_url, timeout=60)
                                if dl_alt.status_code == 200:
                                    with open(output_path, 'wb') as out_f:
                                        out_f.write(dl_alt.content)
                                    print("RapidAPI (free) primary returned image_url; downloaded and saved to output_path")
                                    return output_path
                                else:
                                    print(f"Failed to download RapidAPI (free) primary image_url: {dl_alt.status_code}")
                            except Exception as exd:
                                print(f"Error downloading RapidAPI (free) primary image_url: {exd}")

                    # If we reach here the primary attempt didn't yield an image; log resp.text for diagnosis
                    try:
                        print(f"RapidAPI (free) primary resp.text: {resp_alt.text}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"RapidAPI (free) primary attempt failed: {e}")

                # Next, try the free RapidAPI provider with base64 payload (form-urlencoded)
                try:
                    alt_host_primary = "faceswap-image-transformation-api-free-api-face-swap.p.rapidapi.com"
                    alt_url_primary = f"https://{alt_host_primary}/api/face-swap/create"
                    headers_base64 = {
                        "x-rapidapi-key": rapidapi_key,
                        "x-rapidapi-host": alt_host_primary,
                        "Content-Type": "application/x-www-form-urlencoded"
                    }
                    print(f"Attempting RapidAPI (free) base64 form call at {alt_url_primary}")
                    payload_base64 = {
                        'source': src_b64,
                        'target': tgt_b64
                    }
                    resp_b64 = requests.post(alt_url_primary, data=payload_base64, headers=headers_base64, timeout=90)
                    print(f"RapidAPI (free) base64 response status: {resp_b64.status_code}")
                    try:
                        j_b64 = resp_b64.json()
                    except Exception:
                        j_b64 = None

                    if j_b64:
                        print(f"RapidAPI (free) base64 response keys: {list(j_b64.keys())}")
                        data_block = j_b64.get('data') or {}
                        image_url_b64 = data_block.get('image_url') if isinstance(data_block, dict) else None
                        if isinstance(image_url_b64, str) and image_url_b64.startswith('http'):
                            dl_b64 = requests.get(image_url_b64, timeout=60)
                            if dl_b64.status_code == 200:
                                with open(output_path, 'wb') as out_f:
                                    out_f.write(dl_b64.content)
                                print("RapidAPI (free) base64 returned image_url; downloaded and saved to output_path")
                                return output_path
                            else:
                                print(f"Failed to download RapidAPI (free) base64 image_url: {dl_b64.status_code}")
                    else:
                        try:
                            print(f"RapidAPI (free) base64 resp.text: {resp_b64.text}")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"RapidAPI (free) base64 attempt failed: {e}")

                # If base64 form didn't succeed, fall back to provider-specific faceswapbase64 (if present)
                rapid_url = f"https://{rapidapi_host}/faceswapbase64"
                headers = {
                    "x-rapidapi-key": rapidapi_key,
                    "x-rapidapi-host": rapidapi_host,
                    "Content-Type": "application/json"
                }
                print(f"Attempting RapidAPI faceswap at {rapid_url}")
                # Determine whether to ask the API to match gender.
                # Prefer the explicit character gender saved in app_state; if not set,
                # fall back to a simple heuristic using the character physical description.
                match_gender = False
                try:
                    g = getattr(app_state, 'gender', None)
                    if isinstance(g, str) and g.strip().lower() not in ('', 'unknown', 'unspecified', 'ambiguous'):
                        match_gender = True
                    else:
                        # Fallback: detect gender from description
                        desc_gender = detect_gender_from_description(getattr(app_state, 'physical_description', ''))
                        if desc_gender in ('female', 'male'):
                            match_gender = True
                except Exception:
                    match_gender = False

                payload = {
                    "TargetImageBase64Data": tgt_b64,
                    "SourceImageBase64Data": src_b64,
                    "MatchGender": match_gender,
                    "MaximumFaceSwapNumber": 0
                }
                try:
                    resp = requests.post(rapid_url, json=payload, headers=headers, timeout=60)
                    print(f"RapidAPI response status: {resp.status_code}")
                    # Attempt to parse JSON even if status != 200 since some providers return info there
                    try:
                        j = resp.json()
                    except Exception:
                        j = None

                    # If JSON contains a result URL or embedded data, handle it
                    if j:
                        print(f"RapidAPI response keys: {list(j.keys())}")
                        # Log Success/Message/ProcessingTime if provided
                        success_flag = j.get("Success") if isinstance(j.get("Success"), bool) else (str(j.get("Success", "")).lower() == "true")
                        message = j.get("Message") or j.get("message") or ""
                        proc_time = j.get("ProcessingTime") or j.get("processing_time") or ""
                        print(f"RapidAPI Success: {success_flag}; Message: {message}; ProcessingTime: {proc_time}")

                        # Prefer ResultImageUrl per docs
                        result_url = j.get("ResultImageUrl") or j.get("result_url") or j.get("result")
                        if isinstance(result_url, str) and result_url.strip():
                            # If result_url is a data URI with base64
                            if result_url.startswith("data:"):
                                data = result_url.split(",", 1)[1]
                                img_bytes = base64.b64decode(data)
                                with open(output_path, 'wb') as out_f:
                                    out_f.write(img_bytes)
                                print("RapidAPI returned embedded base64 image; saved to output_path")
                                return output_path

                            # If API returned a URL to download
                            if result_url.startswith("http"):
                                try:
                                    dl = requests.get(result_url, timeout=60)
                                    if dl.status_code == 200:
                                        with open(output_path, 'wb') as out_f:
                                            out_f.write(dl.content)
                                        print("RapidAPI returned image URL; downloaded and saved to output_path")
                                        return output_path
                                    else:
                                        print(f"Failed to download RapidAPI result URL: {dl.status_code}")
                                except Exception as exdl:
                                    print(f"Error downloading RapidAPI result URL: {exdl}")

                        # Some providers return base64 in a different key; try common keys
                        for key in ("ResultImageBase64", "ResultBase64", "result_base64", "image_base64"):
                            val = j.get(key)
                            if isinstance(val, str) and val.strip():
                                try:
                                    img_bytes = base64.b64decode(val)
                                    with open(output_path, 'wb') as out_f:
                                        out_f.write(img_bytes)
                                    print(f"RapidAPI returned base64 via key {key}; saved to output_path")
                                    return output_path
                                except Exception:
                                    pass

                        # If Success==True but no image returned, try multipart /faceswap endpoint before falling back
                        if success_flag:
                            print("RapidAPI reported Success but did not provide ResultImageUrl or embedded image; attempting multipart /faceswap before falling back")
                        else:
                            print("RapidAPI did not succeed or did not return a usable image; attempting multipart /faceswap before falling back")

                        # --- Try alternative RapidAPI provider (free face-swap) as a fallback ---
                        try:
                            alt_host = "faceswap-image-transformation-api-free-api-face-swap.p.rapidapi.com"
                            alt_url = f"https://{alt_host}/api/face-swap/create"
                            headers2 = {
                                "x-rapidapi-key": rapidapi_key,
                                "x-rapidapi-host": alt_host,
                                "Content-Type": "application/x-www-form-urlencoded"
                            }
                            print(f"Attempting RapidAPI (free) faceswap at {alt_url}")

                            # Try sending base64 payload as form-urlencoded fields 'source' and 'target'
                            payload = {
                                'source': src_b64,
                                'target': tgt_b64
                            }
                            resp2 = requests.post(alt_url, data=payload, headers=headers2, timeout=90)

                            print(f"RapidAPI (free) response status: {resp2.status_code}")
                            try:
                                j2 = resp2.json()
                            except Exception:
                                j2 = None

                            if j2:
                                print(f"RapidAPI (free) response keys: {list(j2.keys())}")
                                # Expect structure: {"code":200,"data":{"image_url":"https://..."},"message":""}
                                code = j2.get('code')
                                data_block = j2.get('data') or {}
                                image_url = None
                                if isinstance(data_block, dict):
                                    image_url = data_block.get('image_url') or data_block.get('url')

                                if isinstance(image_url, str) and image_url.startswith('http'):
                                    try:
                                        dl2 = requests.get(image_url, timeout=60)
                                        if dl2.status_code == 200:
                                            with open(output_path, 'wb') as out_f:
                                                out_f.write(dl2.content)
                                            print("RapidAPI (free) returned image_url; downloaded and saved to output_path")
                                            return output_path
                                        else:
                                            print(f"Failed to download RapidAPI (free) image_url: {dl2.status_code}")
                                    except Exception as exd:
                                        print(f"Error downloading RapidAPI (free) image_url: {exd}")

                                # Check for embedded/base64 image fields in top-level or data block
                                for key in ("image_base64", "result", "result_base64", "image", "img_base64"):
                                    val = j2.get(key) or (data_block.get(key) if isinstance(data_block, dict) else None)
                                    if isinstance(val, str) and val.strip():
                                        try:
                                            img_bytes = base64.b64decode(val)
                                            with open(output_path, 'wb') as out_f:
                                                out_f.write(img_bytes)
                                            print(f"RapidAPI (free) returned base64 via key {key}; saved to output_path")
                                            return output_path
                                        except Exception:
                                            pass

                            # If API returned raw image bytes (no JSON) and status==200, save content
                            if resp2.status_code == 200 and (resp2.headers.get('content-type', '').startswith('image') or not j2):
                                try:
                                    with open(output_path, 'wb') as out_f:
                                        out_f.write(resp2.content)
                                    print("RapidAPI (free) returned raw image content; saved to output_path")
                                    return output_path
                                except Exception as exw:
                                    print(f"Failed to write RapidAPI (free) image content: {exw}")

                            print("RapidAPI (free) attempt did not yield an image; falling back to URL-based transfer/PUBLIC_BASE_URL approach")
                        except Exception as e:
                            print(f"RapidAPI (free) attempt failed: {e}")

                        # --- Try URL-based approach: upload files to transfer.sh and call /faceswap with URLs ---
                            try:
                                def upload_to_transfersh(path):
                                    try:
                                        name = os.path.basename(path)
                                        with open(path, 'rb') as upf:
                                            resp_up = requests.put(f'https://transfer.sh/{name}', data=upf, timeout=120)
                                        if resp_up.status_code in (200, 201):
                                            url = resp_up.text.strip()
                                            print(f"Uploaded {name} to transfer.sh -> {url}")
                                            return url
                                        else:
                                            print(f"transfer.sh upload failed for {name}: {resp_up.status_code} {resp_up.text}")
                                            return None
                                    except Exception as upe:
                                        print(f"transfer.sh upload exception for {path}: {upe}")
                                        return None

                                src_url = upload_to_transfersh(template_path)
                                tgt_url = upload_to_transfersh(merge_image_path)
                                if src_url and tgt_url:
                                    try:
                                        # Try free provider for URL-based using JSON then form then multipart text parts
                                        alt_host_primary = "faceswap-image-transformation-api-free-api-face-swap.p.rapidapi.com"
                                        alt_url_primary = f"https://{alt_host_primary}/api/face-swap/create"
                                        headers3 = {
                                            "x-rapidapi-key": rapidapi_key,
                                            "x-rapidapi-host": alt_host_primary,
                                            "Content-Type": "application/json"
                                        }
                                        print(f"Attempting RapidAPI (free) URL-based faceswap at {alt_url_primary} with target={tgt_url} source={src_url}")

                                        # 1) Try JSON body with source_image/target_image
                                        try:
                                            payload3 = {
                                                'source_image': src_url,
                                                'target_image': tgt_url
                                            }
                                            resp3 = requests.post(alt_url_primary, json=payload3, headers=headers3, timeout=120)
                                            print(f"RapidAPI (free) URL-based (json) response status: {resp3.status_code}")
                                            if resp3.status_code == 200:
                                                try:
                                                    j3 = resp3.json()
                                                except Exception:
                                                    j3 = None
                                                if j3 and isinstance(j3.get('data'), dict) and j3['data'].get('image_url'):
                                                    dl3 = requests.get(j3['data']['image_url'], timeout=120)
                                                    if dl3.status_code == 200:
                                                        with open(output_path, 'wb') as out_f:
                                                            out_f.write(dl3.content)
                                                        print("RapidAPI (free) URL-based returned image_url; downloaded and saved to output_path")
                                                        return output_path
                                            else:
                                                try:
                                                    print(f"RapidAPI (free) URL-based (json) resp.text: {resp3.text}")
                                                except Exception:
                                                    pass
                                        except Exception as exj:
                                            print(f"Error calling RapidAPI (free) URL-based (json): {exj}")

                                        # 2) Try form-urlencoded with source_image/target_image
                                        try:
                                            headers_form = {
                                                'x-rapidapi-key': rapidapi_key,
                                                'x-rapidapi-host': alt_host_primary,
                                                'Content-Type': 'application/x-www-form-urlencoded'
                                            }
                                            data_form = {'source_image': src_url, 'target_image': tgt_url}
                                            resp3f = requests.post(alt_url_primary, data=data_form, headers=headers_form, timeout=120)
                                            print(f"RapidAPI (free) URL-based (form) response status: {resp3f.status_code}")
                                            try:
                                                j3f = resp3f.json()
                                            except Exception:
                                                j3f = None
                                            if j3f and isinstance(j3f.get('data'), dict) and j3f['data'].get('image_url'):
                                                dl3f = requests.get(j3f['data']['image_url'], timeout=120)
                                                if dl3f.status_code == 200:
                                                    with open(output_path, 'wb') as out_f:
                                                        out_f.write(dl3f.content)
                                                    print("RapidAPI (free) URL-based (form) returned image_url; downloaded and saved to output_path")
                                                    return output_path
                                            else:
                                                try:
                                                    print(f"RapidAPI (free) URL-based (form) resp.text: {resp3f.text}")
                                                except Exception:
                                                    pass
                                        except Exception as exf:
                                            print(f"Error calling RapidAPI (free) URL-based (form): {exf}")

                                        # 3) Try raw multipart text parts with boundary
                                        try:
                                            boundary = '---011000010111000001101001'
                                            parts = []
                                            parts.append(f"--{boundary}\r\n".encode('utf-8'))
                                            parts.append(b'Content-Disposition: form-data; name="source_image"\r\n\r\n')
                                            parts.append(src_url.encode('utf-8') if isinstance(src_url, str) else str(src_url).encode('utf-8'))
                                            parts.append(b"\r\n")
                                            parts.append(f"--{boundary}\r\n".encode('utf-8'))
                                            parts.append(b'Content-Disposition: form-data; name="target_image"\r\n\r\n')
                                            parts.append(tgt_url.encode('utf-8') if isinstance(tgt_url, str) else str(tgt_url).encode('utf-8'))
                                            parts.append(b"\r\n")
                                            parts.append(f"--{boundary}--\r\n".encode('utf-8'))
                                            body = b''.join(parts)
                                            headers_m = {
                                                'x-rapidapi-key': rapidapi_key,
                                                'x-rapidapi-host': alt_host_primary,
                                                'Content-Type': f'multipart/form-data; boundary={boundary}'
                                            }
                                            resp3m = requests.post(alt_url_primary, headers=headers_m, data=body, timeout=120)
                                            print(f"RapidAPI (free) URL-based (multipart-text) response status: {resp3m.status_code}")
                                            try:
                                                j3m = resp3m.json()
                                            except Exception:
                                                j3m = None
                                            if j3m and isinstance(j3m.get('data'), dict) and j3m['data'].get('image_url'):
                                                dl3m = requests.get(j3m['data']['image_url'], timeout=120)
                                                if dl3m.status_code == 200:
                                                    with open(output_path, 'wb') as out_f:
                                                        out_f.write(dl3m.content)
                                                    print("RapidAPI (free) URL-based (multipart-text) returned image_url; downloaded and saved to output_path")
                                                    return output_path
                                            else:
                                                try:
                                                    print(f"RapidAPI (free) URL-based (multipart-text) resp.text: {resp3m.text}")
                                                except Exception:
                                                    pass
                                        except Exception as exm:
                                            print(f"Error calling RapidAPI (free) URL-based (multipart-text): {exm}")
                                        print(f"RapidAPI URL-based response status: {resp3.status_code}")
                                        if resp3.status_code != 200:
                                            try:
                                                print(f"RapidAPI URL-based resp.text: {resp3.text}")
                                            except Exception:
                                                pass
                                        try:
                                            j3 = resp3.json()
                                        except Exception:
                                            j3 = None

                                        if j3:
                                            print(f"RapidAPI URL-based response keys: {list(j3.keys())}")
                                            # Respect Success and ResultImageUrl
                                            success3 = j3.get("Success") if isinstance(j3.get("Success"), bool) else (str(j3.get("Success", "")).lower() == "true")
                                            msg3 = j3.get("Message") or ""
                                            print(f"RapidAPI URL-based Success: {success3}; Message: {msg3}")
                                            result_url3 = j3.get("ResultImageUrl")
                                            if isinstance(result_url3, str) and result_url3.startswith("http"):
                                                dl3 = requests.get(result_url3, timeout=120)
                                                if dl3.status_code == 200:
                                                    with open(output_path, 'wb') as out_f:
                                                        out_f.write(dl3.content)
                                                    print("RapidAPI URL-based returned image URL; downloaded and saved to output_path")
                                                    return output_path
                                                else:
                                                    print(f"Failed to download RapidAPI URL-based result URL: {dl3.status_code}")
                                        else:
                                            print("RapidAPI URL-based did not return JSON")
                                    except Exception as exu:
                                        print(f"Error calling RapidAPI URL-based faceswap: {exu}")
                                else:
                                    print("Could not upload files to transfer.sh; attempting PUBLIC_BASE_URL fallback if available")
                                    public_base = os.getenv("PUBLIC_BASE_URL", "").rstrip('/')
                                    if public_base:
                                        try:
                                            # Copy files into OUTPUT_DIR so they are served under /images
                                            pub_src_name = f"public_src_{uuid.uuid4()}.jpg"
                                            pub_tgt_name = f"public_tgt_{uuid.uuid4()}.jpg"
                                            pub_src_path = os.path.join(OUTPUT_DIR, pub_src_name)
                                            pub_tgt_path = os.path.join(OUTPUT_DIR, pub_tgt_name)
                                            shutil.copy(template_path, pub_src_path)
                                            shutil.copy(merge_image_path, pub_tgt_path)
                                            src_url = f"{public_base}/images/{pub_src_name}"
                                            tgt_url = f"{public_base}/images/{pub_tgt_name}"
                                            print(f"Using PUBLIC_BASE_URL for RapidAPI: source={src_url} target={tgt_url}")
                                            try:
                                                rapid_url3 = f"https://{rapidapi_host}/faceswap"
                                                headers3 = {
                                                    "x-rapidapi-key": rapidapi_key,
                                                    "x-rapidapi-host": rapidapi_host,
                                                    "Content-Type": "application/json"
                                                }
                                                payload3 = {
                                                    "TargetImageUrl": tgt_url,
                                                    "SourceImageUrl": src_url,
                                                    "MatchGender": match_gender,
                                                    "MaximumFaceSwapNumber": 0
                                                }
                                                print(f"Attempting RapidAPI URL-based faceswap at {rapid_url3} with target={tgt_url} source={src_url}")
                                                resp3 = requests.post(rapid_url3, json=payload3, headers=headers3, timeout=120)
                                                print(f"RapidAPI URL-based response status: {resp3.status_code}")
                                                if resp3.status_code != 200:
                                                    try:
                                                        print(f"RapidAPI URL-based resp.text: {resp3.text}")
                                                    except Exception:
                                                        pass
                                                try:
                                                    j3 = resp3.json()
                                                except Exception:
                                                    j3 = None

                                                if j3:
                                                    print(f"RapidAPI URL-based response keys: {list(j3.keys())}")
                                                    success3 = j3.get("Success") if isinstance(j3.get("Success"), bool) else (str(j3.get("Success", "")).lower() == "true")
                                                    msg3 = j3.get("Message") or ""
                                                    print(f"RapidAPI URL-based Success: {success3}; Message: {msg3}")
                                                    result_url3 = j3.get("ResultImageUrl")
                                                    if isinstance(result_url3, str) and result_url3.startswith("http"):
                                                        dl3 = requests.get(result_url3, timeout=120)
                                                        if dl3.status_code == 200:
                                                            with open(output_path, 'wb') as out_f:
                                                                out_f.write(dl3.content)
                                                            print("RapidAPI URL-based returned image URL; downloaded and saved to output_path")
                                                            return output_path
                                                        else:
                                                            print(f"Failed to download RapidAPI URL-based result URL: {dl3.status_code}")
                                                else:
                                                    print("RapidAPI URL-based did not return JSON")
                                            except Exception as exu:
                                                print(f"Error calling RapidAPI URL-based faceswap: {exu}")
                                        except Exception as cpe:
                                            print(f"PUBLIC_BASE_URL fallback failed to prepare public files: {cpe}")
                                    else:
                                        print("PUBLIC_BASE_URL not configured; skipping URL-based RapidAPI call")
                            except Exception as ex_all:
                                print(f"URL-based RapidAPI fallback failed: {ex_all}")
                        except Exception as e:
                            print(f"RapidAPI multipart faceswap attempt failed: {e}")
                except Exception as e:
                    # RapidAPI request failed; we'll fall back to Face++ below
                    print(f"RapidAPI faceswap attempt failed with exception: {e}")

        except Exception:
            # non-fatal - RapidAPI failed, continue to Hugging Face fallback
            pass

        # RapidAPI failed, try Hugging Face fallback directly
        print("RapidAPI failed, attempting Hugging Face fallback...")
        if huggingface_face_swap(template_path, target_path, output_path):
            print("Hugging Face fallback succeeded")
            return output_path
        else:
            print("Hugging Face fallback also failed, returning original image")
            # If all face swap methods fail, return the original target image
            try:
                shutil.copy(target_path, output_path)
                return output_path
            except Exception:
                return None

    except Exception as e:
        print(f"Face swap failed: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Try Hugging Face fallback as last resort
        print("Attempting Hugging Face fallback as last resort...")
        if huggingface_face_swap(template_path, target_path, output_path):
            print("Hugging Face fallback succeeded")
            return output_path
        else:
            print("Hugging Face fallback also failed, returning original image")
            # If all face swap methods fail, return the original target image
            try:
                shutil.copy(target_path, output_path)
                return output_path
            except Exception:
                return None


def detect_gender_from_description(description):
    """Extract gender information from character description"""
    if not description:
        return "unknown"
    
    description = description.lower()
    
    # Female indicators
    female_terms = ["woman", "lady", "girl", "female", "feminine", "she", "her", "actress", 
                    "princess", "queen", "waitress", "wife", "mother", "daughter", "sister"]
    
    # Male indicators
    male_terms = ["man", "guy", "boy", "male", "masculine", "he", "his", "actor", 
                  "prince", "king", "waiter", "husband", "father", "son", "brother"]
    
    # Count occurrences of gender terms
    female_count = sum(1 for term in female_terms if term in description)
    male_count = sum(1 for term in male_terms if term in description)
    
    # Determine gender based on term frequency
    if female_count > male_count:
        return "female"
    elif male_count > female_count:
        return "male"
    else:
        # Check for specific pronouns which are strong indicators
        if "she" in description or "her" in description:
            return "female"
        elif "he" in description or "his" in description:
            return "male"
        else:
            return "unknown"





def detect_face_gender(face):
    """Detect gender from a face object using simple heuristics"""
    # This is a simplified approach - in a production system, you'd use a proper gender classifier
    try:
        print(f"Detecting gender for face with attributes: {dir(face)}")
        
        # Check if the face has gender attribute (some insightface models provide this)
        if hasattr(face, 'gender') and face.gender is not None:
            gender = "female" if face.gender == 0 else "male"
            print(f"Gender detected from face attribute: {gender} (value: {face.gender})")
            return gender
        
        # Fallback to simple heuristics based on face shape
        # These are very rough approximations and not scientifically accurate
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106
            
            # Calculate face width to height ratio
            face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
            face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            ratio = face_width / face_height if face_height > 0 else 0
            print(f"Face width/height ratio: {ratio:.2f}")
            
            # Very rough heuristic - women tend to have slightly more oval faces
            # This is NOT accurate and should be replaced with a proper classifier
            if ratio < 0.85:
                print(f"Detected as female based on face ratio {ratio:.2f} < 0.85")
                return "female"
            elif ratio > 0.95:
                print(f"Detected as male based on face ratio {ratio:.2f} > 0.95")
                return "male"
            else:
                print(f"Face ratio {ratio:.2f} is in ambiguous range (0.85-0.95)")
        else:
            print("No facial landmarks available for gender detection")
        
        # If we can't determine, return unknown
        print("Unable to determine gender from face attributes, returning 'unknown'")
        return "unknown"
    except Exception as e:
        print(f"Error in gender detection: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return "unknown"


# =====================
# Image Generation
# =====================

def generate_image(prompt, seed=None):
    """Generate an image using ImageRouter API and return (filename, message)"""
    try:
        print("\n==== IMAGE GENERATION (ImageRouter) STARTED =====")
        print(f"INITIAL PROMPT: {prompt}")
        print(f"SEED VALUE: {seed if seed is not None else 'None (random)'}")
        print("====================================")

        # Check cache (use prompt+seed as key)
        cache_key = f"{prompt}_{seed}" if seed is not None else prompt
        if cache_key in app_state.prompt_cache:
            cached = app_state.prompt_cache[cache_key]
            cached_path = os.path.join(OUTPUT_DIR, cached) if not os.path.isabs(cached) else cached
            if os.path.exists(cached_path):
                print("Cache hit: returning previously generated image")
                return os.path.basename(cached), "Retrieved from cache"

        # Prepare ImageRouter request
        api_key = app_state.imagerouter_api_key or os.getenv("IMAGEROUTER_API_KEY", "")
        if not api_key:
            print("ERROR: ImageRouter API key not configured")
            return None, "ImageRouter API key not configured"

        url = "https://api.imagerouter.io/v1/openai/images/generations"
        payload = {
            "prompt": prompt,
            # Use HiDream model by default per user request
            "model": app_state.available_models[0] if app_state.available_models else "HiDream-ai/HiDream-I1-Full:free",
            "response_format": "b64_json"
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        print("Sending request to ImageRouter...")
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        # If ImageRouter fails (rate limit 429 or other errors), fall back to AI Horde
        if resp.status_code != 200:
            print(f"ImageRouter returned status {resp.status_code}: {resp.text}")
            # Try to detect rate limit / daily limit message
            text_lower = resp.text.lower() if resp.text else ""
            try:
                j = resp.json()
            except Exception:
                j = {}

            is_rate_limit = False
            # Common ImageRouter/Stripe/OpenAI style rate limit messages
            if resp.status_code == 429:
                is_rate_limit = True
            if isinstance(j, dict) and j.get("error"):
                # e.g. {"error": {"message": "Daily limit of 50 free requests reached...", "type": "rate_limit_error"}}
                err = j.get("error")
                if isinstance(err, dict) and ("rate_limit" in str(err.get("type", "")).lower() or "daily limit" in str(err.get("message", "")).lower()):
                    is_rate_limit = True

            if is_rate_limit:
                print("Detected ImageRouter rate limit — attempting AI Horde fallback")
                aihorde_filename, aihorde_msg = generate_image_ai_horde(prompt, seed=seed)
                if aihorde_filename:
                    return aihorde_filename, f"AI Horde fallback: {aihorde_msg}"
                else:
                    return None, f"ImageRouter error {resp.status_code}: {resp.text} — AI Horde fallback failed: {aihorde_msg}"

            return None, f"ImageRouter error {resp.status_code}: {resp.text}"

        data = resp.json()
        # Extract image data
        image_bytes = None
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            first = data["data"][0]
            if "b64_json" in first:
                image_bytes = base64.b64decode(first["b64_json"])
            elif "url" in first:
                image_url = first["url"]
                print(f"ImageRouter returned URL: {image_url}, downloading...")
                dl = requests.get(image_url, timeout=30)
                if dl.status_code == 200:
                    image_bytes = dl.content
                else:
                    return None, f"Failed to download image URL: {dl.status_code}"

        if not image_bytes:
            print("No image found in ImageRouter response")
            return None, "No image in ImageRouter response"

        # Save image to output dir
        output_filename = f"generated_{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        # Cache and return
        app_state.prompt_cache[cache_key] = output_filename
        app_state.last_used_prompt = prompt
        print(f"Saved image to {output_path}")
        return output_filename, "Image generated successfully"
    except Exception as e:
        print("\n==== IMAGE GENERATION ERROR =====")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        print("==================================")
        # Attempt AI Horde fallback on unexpected exceptions as well
        try:
            print("Attempting AI Horde fallback after exception")
            aihorde_filename, aihorde_msg = generate_image_ai_horde(prompt, seed=seed)
            if aihorde_filename:
                return aihorde_filename, f"AI Horde fallback after exception: {aihorde_msg}"
            else:
                return None, f"Error generating image: {str(e)}; AI Horde fallback failed: {aihorde_msg}"
        except Exception as e2:
            print(f"AI Horde fallback also failed: {e2}")
            return None, f"Error generating image: {str(e)}"


def generate_image_ai_horde(prompt, seed=None):
    """Generate an image using AI Horde (Stable Horde) async API as a fallback.

    Returns (filename, message) on success or (None, error_message) on failure.
    """
    try:
        print("\n==== AI HORDE FALLBACK STARTED =====")
        print(f"PROMPT: {prompt}")
        print(f"SEED: {seed if seed is not None else 'None'}")
        API_URL = "https://aihorde.net/api/v2/generate/async"
        CHECK_URL = "https://aihorde.net/api/v2/generate/check"

        api_key = os.getenv("AIHORDE_API_KEY", "")
        headers = {"apikey": api_key} if api_key else {}

        data = {
            "prompt": prompt,
            "models": ["juggernaut_xl"],  # AI Horde expects "models" array
            "params": {
                "width": 512,
                "height": 768,
                "steps": 30,
                "cfg_scale": 7.5,
                "sampler_name": "k_euler_a",
                "clip_skip": 1,
                "nsfw": True
            }
        }
        
        # Add seed if provided
        if seed is not None:
            data["params"]["seed"] = str(seed)

        print("Sending request to AI Horde...")
        resp = requests.post(API_URL, json=data, headers=headers, timeout=30)
        if resp.status_code not in (200, 201, 202):
            print(f"AI Horde returned status {resp.status_code}: {resp.text}")
            return None, f"AI Horde start error {resp.status_code}: {resp.text}"

        job = resp.json()
        job_id = job.get("id") or job.get("job")
        if not job_id:
            print(f"AI Horde did not return job id: {job}")
            return None, "AI Horde did not return job id"

        print(f"AI Horde job started: {job_id}, polling for result...")

        # Poll for completion
        max_attempts = 120
        attempt = 0
        img_b64 = None
        while attempt < max_attempts:
            attempt += 1
            try:
                check_resp = requests.get(f"{CHECK_URL}/{job_id}", headers=headers, timeout=30)
                if check_resp.status_code != 200:
                    print(f"AI Horde check returned {check_resp.status_code}: {check_resp.text}")
                    time.sleep(2)
                    continue
                chk = check_resp.json()
                print(f"AI Horde status check {attempt}: done={chk.get('done')}, finished={chk.get('finished', 0)}, processing={chk.get('processing', 0)}, waiting={chk.get('waiting', 0)}")
                
                if chk.get("done"):
                    gens = chk.get("generations") or []
                    if isinstance(gens, list) and len(gens) > 0:
                        first = gens[0]
                        img_b64 = first.get("img") or first.get("base64") or first.get("b64")
                        if img_b64:
                            break
                    print(f"Request done but no image found in generations: {gens}")
                    return None, "AI Horde generation completed but no image returned"
                
                # Check for faulted/failed status
                if chk.get("faulted"):
                    return None, "AI Horde generation failed (faulted)"
                    
                # Not done yet, continue polling
                time.sleep(2)
            except Exception as e:
                print(f"AI Horde poll error: {e}")
                time.sleep(2)

        if not img_b64:
            print("AI Horde did not return an image within timeout or attempts")
            return None, "AI Horde generation timed out or failed"

        print(f"AI Horde returned image data (length: {len(img_b64) if isinstance(img_b64, str) else 'not string'})")

        # img_b64 should be a base64 string
        if isinstance(img_b64, str):
            try:
                # Handle data URLs like "data:image/webp;base64,..."
                if img_b64.startswith("data:"):
                    comma = img_b64.find(",")
                    if comma != -1:
                        img_b64 = img_b64[comma+1:]
                image_bytes = base64.b64decode(img_b64)
            except Exception as e:
                print(f"Failed to decode base64 image: {e}")
                return None, "AI Horde returned invalid base64 image data"
        else:
            print("AI Horde returned non-string image data")
            return None, "AI Horde returned unexpected image data format"

        # Save image
        output_filename = f"generated_aihorde_{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        # Cache and set last prompt
        cache_key = f"{prompt}_{seed}" if seed is not None else prompt
        app_state.prompt_cache[cache_key] = output_filename
        app_state.last_used_prompt = prompt
        print(f"AI Horde saved image to {output_path}")
        return output_filename, "AI Horde image generated successfully"
    except Exception as e:
        print(f"AI Horde fallback error: {e}")
        import traceback
        print(traceback.format_exc())
        return None, f"AI Horde error: {str(e)}"


def generate_image_with_face_swap(response_text, seed=None):
    try:
        # ===== DETAILED LOGGING: Face Swap Process Started =====
        print("\n==== FACE SWAP PROCESS STARTED =====")
        print(f"ORIGINAL PROMPT: {response_text}")
        print(f"REQUESTED SEED: {seed if seed is not None else 'None (random)'}")
        print("======================================")
        
        # Clean and normalize the response text for checking
        cleaned_text = response_text.strip()
        cleaned_text = re.sub(r'^\*+\s*', '', cleaned_text)  # Remove leading asterisks and spaces
        cleaned_text = re.sub(r'\s*\*+$', '', cleaned_text)  # Remove trailing asterisks and spaces
        cleaned_text = cleaned_text.strip().lower()
        
        if cleaned_text == "none":
            print("\n==== FACE SWAP SKIPPED =====")
            print(f"REASON: Prompt is 'none' (original: '{response_text}')")
            print("=============================")
            return None, "Image prompt was 'none' — skipping image generation"

        # If no character face was uploaded, we will still generate a base image
        # from the prompt and skip the face-swap step. This ensures images are
        # produced even if the user hasn't uploaded a face yet.
        do_face_swap = True
        if not app_state.face_image_path and not app_state.source_img:
            print("No uploaded character face found — will generate base image without face swap")
            do_face_swap = False
            
        # For the first image in chat, use the saved character seed if available and no specific seed was provided
        if seed is None and len(app_state.chat_history) <= 1 and app_state.character_seed is not None:
            # ===== DETAILED LOGGING: Character Seed Usage =====
            print("\n==== CHARACTER SEED USAGE =====")
            print(f"USING CHARACTER SEED: {app_state.character_seed}")
            print(f"REASON: First chat message and no specific seed provided")
            print("================================")
            seed = app_state.character_seed

        # ===== DETAILED LOGGING: Base Image Generation =====
        print("\n==== STARTING BASE IMAGE GENERATION =====")
        print(f"PROMPT: {response_text}")
        print(f"SEED: {seed if seed is not None else 'None (random)'}")
        print("==========================================")
        
        # Generate the image
        image_filename, message = generate_image(response_text, seed)
        if not image_filename:
            print("\n==== FACE SWAP ABORTED =====")
            print(f"REASON: Base image generation failed - {message}")
            print("============================")
            return None, message

        # ===== DETAILED LOGGING: Face Swap Preparation =====
        print("\n==== FACE SWAP PREPARATION =====")
        print(f"BASE IMAGE FILENAME: {image_filename}")
        
        # Prepare paths for face swapping
        # Ensure we have the full path to the generated image
        generated_path = os.path.join(OUTPUT_DIR, image_filename)
        print(f"FULL BASE IMAGE PATH: {generated_path}")
        
        if not os.path.exists(generated_path):
            print(f"WARNING: Generated image not found at expected path")
            # Check if the image_filename is already a full path
            if os.path.exists(image_filename):
                generated_path = image_filename
                print(f"FOUND IMAGE AT ALTERNATE PATH: {generated_path}")
            else:
                print(f"ERROR: Image not found at any path")
                print("=================================")
                return None, f"Generated image not found at {generated_path}"

        swapped_filename = f"swapped_{uuid.uuid4()}.jpg"
        swapped_path = os.path.join(OUTPUT_DIR, swapped_filename)
        print(f"TARGET SWAP PATH: {swapped_path}")
        print("================================")

        # ===== DETAILED LOGGING: Face Swap Execution =====
        print("\n==== EXECUTING FACE SWAP =====")
        print(f"SOURCE FACE: {'Available' if app_state.source_face is not None else 'Missing'}")
        print(f"SOURCE IMAGE: {'Available' if app_state.source_img is not None else 'Missing'}")
        print(f"TARGET IMAGE: {generated_path}")
        print(f"OUTPUT PATH: {swapped_path}")
        
        result_path = None
        if do_face_swap:
            # Perform face swapping
            result_path = swap_face(app_state.source_face, app_state.source_img, generated_path, swapped_path)

            if not result_path:
                print("FACE SWAP RESULT: Failed")
                print("FALLBACK: Using original image")
                print("==============================")
                # Return the base generated image if swapping fails
                app_state.last_used_prompt = response_text
                return image_filename, "Face swap failed, using original image"

            print(f"FACE SWAP RESULT: Success")
            print(f"RESULT PATH: {result_path}")
            print("==============================")

            # Store the last used prompt
            app_state.last_used_prompt = response_text
            print("\n==== FACE SWAP COMPLETE =====")
            print(f"FINAL IMAGE: {os.path.basename(result_path)}")
            print(f"PROMPT SAVED FOR REGENERATION: {response_text[:50]}{'...' if len(response_text) > 50 else ''}")
            print("=============================")

            return os.path.basename(result_path), "Image generated and face-swapped successfully"
        else:
            # No face uploaded - use the generated base image
            print("Skipping face swap; returning base generated image")
            app_state.last_used_prompt = response_text
            print("==== FACE SWAP COMPLETE (SKIPPED) =====")
            print(f"FINAL IMAGE: {image_filename}")
            print("========================================")
            return image_filename, "Generated base image (no face uploaded)"
    except Exception as e:
        # ===== DETAILED LOGGING: Face Swap Error =====
        print("\n==== FACE SWAP ERROR =====")
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        print("==========================")
        return None, f"Error generating image: {str(e)}"


def regenerate_image(seed=None):
    """Regenerate the last image with a new seed"""
    # ===== DETAILED LOGGING: Regeneration Started =====
    print("\n==== IMAGE REGENERATION STARTED =====")
    print(f"LAST USED PROMPT: {app_state.last_used_prompt if app_state.last_used_prompt else 'None'}") 
    print(f"REQUESTED SEED: {seed if seed is not None else 'None (will generate random)'}") 
    print("=====================================")
    
    if not app_state.last_used_prompt:
        print("\n==== REGENERATION ERROR =====")
        print("ERROR: No previous image prompt available")
        print("==============================")
        return None, "No previous image prompt available"

    # Use a new random seed if none provided
    if seed is None:
        seed = int(time.time())
        print("\n==== RANDOM SEED GENERATED =====")
        print(f"GENERATED SEED: {seed}")
        print("================================")
    
    print("\n==== CALLING FACE SWAP WITH PREVIOUS PROMPT =====")
    print(f"PROMPT: {app_state.last_used_prompt}")
    print(f"SEED: {seed}")
    print("===============================================")

    return generate_image_with_face_swap(app_state.last_used_prompt, seed)


# =====================
# Mistral LLM Integration
# =====================
def get_image_context(num_entries=3):
    """Retrieve context from recent image prompts to maintain visual consistency"""
    if not hasattr(app_state, 'image_history') or len(app_state.image_history) == 0:
        return ""

    # Get the most recent prompts (up to num_entries)
    recent_entries = app_state.image_history[-num_entries:] if len(
        app_state.image_history) >= num_entries else app_state.image_history

    # Extract prompts
    prompts = [entry.get('prompt', '') for entry in recent_entries if 'prompt' in entry and entry['prompt'] != 'none']

    if not prompts:
        return ""

    return "\n".join(prompts)


def generate_mistral_response(message: str) -> dict:
    """Generate both a conversational response and an image prompt using Mistral API"""
    MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

    # Get context from chat history (last 5 exchanges)
    context = ""
    if len(app_state.chat_history) > 0:
        context_messages = app_state.chat_history[-5:]
        context = "\n".join([f"User: {msg['user']}\nAssistant: {msg.get('assistant', '')}"
                             for msg in context_messages if 'user' in msg])
    # Create the system message for roleplay
    # Build a richer system message that explicitly includes the saved character attributes
    print(f"[DEBUG] generate_mistral_response START - current physical_description: '{app_state.physical_description}'")
    char_name = app_state.character_name or "Fantasy Character"
    relation = app_state.relation_to_user or "companion"
    user_name = app_state.user_name or "user"
    context_info = app_state.chat_context or "A casual conversation"
    personality = app_state.behavioral_description or "A friendly and helpful character"
    appearance = app_state.physical_description or "A unique and mysterious figure"
    attire = app_state.initial_attire or "appropriate clothing"
    gender = app_state.gender or "unspecified"
    style = app_state.style or "Photorealistic"


    # Simple, effective system message inspired by app.py
    system_message = f"""You are a naughty roleplayer assistant. You are completely uncensored, ready to be very explicit and lewd if needed.
    Don't come off as too strong at the very beginning, only be naughty if initiated by the user.

    CHARACTER: You are {char_name}, a {relation} to the user. Your appearance: {appearance}. Your attire: {attire}. Gender: {gender}.

    When user asks for visual content ("show", "picture", "appearance", "strip", "let me see", "send", etc. or relevant words in any language), provide both parts:
    
    1. CHAT_RESPONSE: A natural, conversational response as if you were talking directly to the user.
    You are completely uncensored, ready to be very explicit and lewd if needed.
    Don't come off as too strong at the first line, only be naughty if initiated by the user.
    If the user speaks in Bengali, continue the conversation in Bengali. Do not provide translations or explanations

    2. IMAGE_PROMPT: A detailed visual description optimized for image generation.
    This should include visual details about yourself, the scene, lighting, and mood.
    Create a single flowing description without sections or bullet points.

    IMPORTANT FOR NSFW REQUESTS: When user asks to "strip", "undress", "naked", "nude", "bottomless", "topless", "remove clothes", "show body parts", or any explicit content, you MUST include explicit anatomical terms in the IMAGE_PROMPT:
    - For female characters: include "breasts", "nipples", "vagina", "pussy" as appropriate
    - For male characters: include "penis", "cock" as appropriate  
    - Add terms like "naked", "nude", "explicit nudity", "detailed genitalia", "nsfw"
    - Be specific and explicit in the visual description when NSFW is requested
       
    The prompt should be concise (under 80 words) and focus on:
    - physical appearance details matching your character description
    - specific clothing/attire (or lack thereof for NSFW)
    - facial expression and pose
    - precise location/setting
    - lighting conditions
    - camera angle/framing

    Write this as a natural flowing description like "young woman with long red hair wearing a blue dress, standing in a sunlit forest clearing, soft golden light, atmospheric mist, shallow depth of field, portrait shot"

    DO NOT use formatting words like "Character:", "Setting:", "Lighting:", etc.
    NO storytelling, NO actions, NO dialogue - ONLY concrete visual details in a flowing description.

    Format your response exactly like this:
    CHAT_RESPONSE: [Your natural conversational response here]
    IMAGE_PROMPT: [Detailed visual description for image generation here]

    Only generate an IMAGE_PROMPT when the conversation calls for showing an image. If no image is needed, respond with "IMAGE_PROMPT: none".

    Stay in character as {char_name}."""

    # Prepare the API request
    headers = {"Authorization": f"Bearer {app_state.mistral_api_key}",
               "Content-Type": "application/json"}

    # Prepare messages
    messages = [{"role": "system", "content": system_message}]
    if context:
        messages.append({"role": "user", "content": f"Previous conversation:\n{context}"})

    # Add image context for consistency if available
    image_context = get_image_context(3)
    if image_context:
        messages.append({"role": "user",
                         "content": f"For visual consistency, these were the previous image descriptions used. Try to maintain consistency with these when generating new image prompts:\n{image_context}"})

    # Simple character reminder
    char_attrs = f"Character: {char_name} - Appearance: {appearance} - Attire: {attire}. Use these EXACT details in IMAGE_PROMPT, not generic terms."
    
    # Debug: log the character attributes being sent
    print(f"[generate_mistral_response] Character attributes sent to Mistral:")
    print(f"  Name: {char_name}")
    print(f"  Appearance: {appearance}")
    print(f"  Attire: {attire}")
    print(f"  Gender: {gender}")
    print(f"  Style: {style}")
    
    # Validation: Check if we have meaningful character data
    has_specific_character = bool(
        app_state.physical_description and 
        app_state.physical_description != "A unique and mysterious figure" and
        len(app_state.physical_description.strip()) > 10
    )
    print(f"[generate_mistral_response] Has specific character data: {has_specific_character}")
    if not has_specific_character:
        print(f"[generate_mistral_response] WARNING: No specific character saved - using defaults!")
    
    messages.append({"role": "user", "content": char_attrs})
    
    messages.append({"role": "user", "content": message})

    payload = {
        # Use the larger model to produce richer prompts (match copy.py)
        "model": "mistral-medium-latest",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800  # Provide enough tokens for both parts
    }

    try:
        response = requests.post(MISTRAL_ENDPOINT,
                                 headers=headers,
                                 json=payload)

        if response.status_code == 200:
            full_response = response.json()["choices"][0]["message"]["content"]
            # Debug: log the full raw assistant content so we can inspect why
            # IMAGE_PROMPT may be suppressed or changed by the model.
            print("--- Mistral full raw response START ---")
            print(full_response)
            print("--- Mistral full raw response END ---")

            # Parse the response to separate chat response and image prompt
            chat_response = ""
            image_prompt = "none"

            # Extract chat response (case-insensitive) and image prompt - handle markdown formatting
            chat_match = re.search(r"\*?\*?chat_response:\*?\*?\s*(.*?)(?=\*?\*?image_prompt:|$)", full_response, re.IGNORECASE | re.DOTALL)
            if chat_match:
                chat_response = chat_match.group(1).strip()

            prompt_match = re.search(r"\*?\*?image_prompt:\*?\*?\s*(.*?)$", full_response, re.IGNORECASE | re.DOTALL)
            if prompt_match:
                image_prompt = prompt_match.group(1).strip()

            # Clean up markdown formatting from responses
            if chat_response:
                chat_response = re.sub(r'^\*?\*?', '', chat_response)  # Remove leading asterisks
                chat_response = re.sub(r'\*?\*?$', '', chat_response)  # Remove trailing asterisks
                chat_response = chat_response.strip()
            
            if image_prompt:
                image_prompt = re.sub(r'^\*?\*?', '', image_prompt)    # Remove leading asterisks
                image_prompt = re.sub(r'\*?\*?$', '', image_prompt)    # Remove trailing asterisks  
                image_prompt = image_prompt.strip()

            # If the assistant didn't follow the exact format, fall back heuristics
            if not chat_response:
                # Try to find a natural-language assistant reply before any 'IMAGE_PROMPT' marker
                parts = re.split(r"\*?\*?image_prompt:\*?\*?\s*", full_response, flags=re.IGNORECASE)
                if len(parts) > 1:
                    # Take everything before the image_prompt marker as chat response
                    chat_response = parts[0].strip()
                    image_prompt = parts[1].strip()
                    # Clean up any remaining markdown
                    chat_response = re.sub(r'^\*?\*?', '', chat_response).strip()
                    image_prompt = re.sub(r'^\*?\*?', '', image_prompt).strip()
                else:
                    # No marker at all — treat full response as chat_response
                    chat_response = full_response.strip()

            # If no image_prompt was explicitly returned, attempt to auto-detect a visual description
            if (not prompt_match) or (not image_prompt) or image_prompt.lower() == "none":
                # Use a simple heuristic: take the shortest last sentence that looks like a visual description
                candidates = [s.strip() for s in re.split(r"\n|\.|;", full_response) if len(s.strip())>10]
                if candidates:
                    # Prefer candidates that contain visual trigger words
                    vis_cands = [c for c in candidates if any(w in c.lower() for w in ["wearing","standing","standing in","portrait","lighting","light","pose","wear","dress","hair","eyes","skin","background","scene"]) ]
                    pick = vis_cands[-1] if vis_cands else candidates[-1]
                    # If pick looks like a full sentence longer than 8 chars, use it as image prompt
                    if len(pick) < 200 and len(pick) > 15:
                        image_prompt = pick

            return {
                "chat_response": chat_response,
                "image_prompt": "none" if (not image_prompt or image_prompt.lower().strip() == "none") else image_prompt
            }
        else:
            error_msg = f"Sorry, I encountered an error: {response.status_code}. Please check your API key."
            return {"chat_response": error_msg, "image_prompt": "none"}
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        return {"chat_response": error_msg, "image_prompt": "none"}


def should_generate_image(response: str) -> bool:
    """Determine if an image should be generated based on the response"""
    # Keywords that suggest visual content
    visual_triggers = [
        "show you", "imagine", "picture", "visualize", "look like",
        "appearance", "see me", "this is how", "visually", "image",
        "here i am", "what i look like", "let me show", "here's what",
        "i appear", "i look", "you'd see"
    ]

    response_lower = response.lower()
    return any(trigger in response_lower for trigger in visual_triggers)

# ✅ Genitalia Enhancement Logic


def extract_camera_angle(user_message: str) -> Optional[str]:
    """Extract a camera angle direction from user message"""
    message = user_message.lower()
    if any(kw in message for kw in ["face me", "from front", "look at me", "face the camera", "turn toward", "full frontal"]):
        return "from front"
    elif any(kw in message for kw in ["turn around", "from behind", "backside", "show me your back", "back view"]):
        return "from behind"
    elif any(kw in message for kw in ["from side", "profile", "side view", "look sideways", "side angle"]):
        return "from side"
    return None

# =====================
# FastAPI Endpoints
# =====================

class ApiSettings(BaseModel):
    mistral_api_key: str
    imagerouter_api_key: Optional[str] = ""
    facepp_api_key: Optional[str] = ""
    facepp_api_secret: Optional[str] = ""
    rapidapi_key: Optional[str] = ""
    rapidapi_host: Optional[str] = ""
    sd_model_path: Optional[str] = ""
    scheduler_type: Optional[str] = "dpm_2m_karras"
    guidance_scale: Optional[float] = 7.0
    num_inference_steps: Optional[int] = 30
    clip_skip: Optional[int] = 2


@app.post("/set_api_settings")
async def set_api_settings(settings: ApiSettings):
    """Set the API keys and model settings"""
    app_state.mistral_api_key = settings.mistral_api_key
    # Store ImageRouter API key if provided
    if getattr(settings, 'imagerouter_api_key', None):
        app_state.imagerouter_api_key = settings.imagerouter_api_key
    # Store Face++ API credentials if provided
    if getattr(settings, 'facepp_api_key', None):
        app_state.facepp_api_key = settings.facepp_api_key
    if getattr(settings, 'facepp_api_secret', None):
        app_state.facepp_api_secret = settings.facepp_api_secret
    # Store RapidAPI key/host if provided (allows runtime update without dyno restart)
    if getattr(settings, 'rapidapi_key', None):
        app_state.rapidapi_key = settings.rapidapi_key
    if getattr(settings, 'rapidapi_host', None):
        app_state.rapidapi_host = settings.rapidapi_host
    
    # Save generation settings
    app_state.scheduler_type = settings.scheduler_type
    app_state.guidance_scale = settings.guidance_scale
    app_state.num_inference_steps = settings.num_inference_steps
    app_state.clip_skip = settings.clip_skip

    # Initialize SD model if path provided
    if settings.sd_model_path:
        app_state.sd_model_path = settings.sd_model_path
        if not app_state.initialize_sd_model():
            return JSONResponse(
                content={"success": False, "message": "Failed to initialize Stable Diffusion model"},
                status_code=400
            )

    # Test the Mistral API key with a simple request
    test_message = "Hello"
    try:
        response = generate_mistral_response(test_message)
        if isinstance(response, dict):
            chat_resp = response.get("chat_response", "")
            if "error" in chat_resp.lower() or "sorry, i encountered an error" in chat_resp.lower():
                return JSONResponse(
                    content={"success": False, "message": "API key test failed: " + chat_resp},
                    status_code=400
                )

        return JSONResponse(
            content={"success": True, "message": "API settings saved successfully!"},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"Error testing API key: {str(e)}"},
            status_code=400
        )


@app.post("/upload_character")
async def upload_character(
        face_image: UploadFile = File(...),
        physical_description: str = Form(...),
        behavioral_description: str = Form(...),
        character_name: str = Form(""),
        relation_to_user: str = Form(""),
        user_name: str = Form(""),
        chat_context: str = Form(""),
        initial_attire: str = Form(""),
        gender: str = Form("Female"),
        style: str = Form("Photorealistic")
):
    """Upload a character face image and description with extended details"""
    try:
        # Save the uploaded face image
        file_location = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{face_image.filename}")
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(face_image.file, buffer)

        # Read the uploaded image (no local face detection). extract_face_from_image
        # returns (None, img_rgb) for compatibility with previous callers.
        _, source_img = extract_face_from_image(file_location)
        if source_img is None:
            return JSONResponse(
                content={"success": False, "message": "Failed to read the uploaded image"},
                status_code=400
            )

        # Store the image and path (no local face object)
        app_state.source_face = None
        app_state.source_img = source_img
        app_state.face_image_path = file_location
        print(f"[DEBUG] Setting physical_description in setup_character: '{physical_description}'")
        app_state.physical_description = physical_description
        app_state.behavioral_description = behavioral_description
        app_state.character_name = character_name
        app_state.relation_to_user = relation_to_user
        app_state.user_name = user_name
        app_state.chat_context = chat_context
        app_state.initial_attire = initial_attire
        app_state.gender = gender
        app_state.style = style
        
        # Preserve existing character base prompt and seed if they exist
        # This ensures we don't lose them when updating character details
        if hasattr(app_state, 'character_base_prompt') and app_state.character_base_prompt:
            print(f"Preserving existing character base prompt: {app_state.character_base_prompt[:30]}...")
        if hasattr(app_state, 'character_seed') and app_state.character_seed:
            print(f"Preserving existing character seed: {app_state.character_seed}")

        # Create a simple face preview image by center-cropping the uploaded image
        h, w = source_img.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h) // 4
        face_crop = source_img[max(0, center_y-size):min(h, center_y+size), max(0, center_x-size):min(w, center_x+size)]

        # If the crop is empty for some reason, fall back to a smaller central crop
        if face_crop is None or getattr(face_crop, 'size', 0) == 0:
            size = min(w, h) // 8
            face_crop = source_img[max(0, center_y-size):min(h, center_y+size), max(0, center_x-size):min(w, center_x+size)]
        
        # Ensure the crop is at least 112x112
        face_crop = cv2.resize(face_crop, (112, 112))

        # Convert NumPy array to PIL Image
        face_crop_pil = Image.fromarray(face_crop)

        # Encode to base64
        buffered = BytesIO()
        face_crop_pil.save(buffered, format="JPEG")
        face_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return JSONResponse(
            content={
                "success": True,
                "message": "Character details saved successfully!",
                "face_data": face_data
            },
            status_code=200
        )
    except Exception as e:
        print(f"Error uploading character: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            content={"success": False, "message": f"Error: {str(e)}"},
            status_code=500
        )

class CharacterPreviewRequest(BaseModel):
    base_prompt: str
    seed: Optional[int] = None

@app.post("/generate_character_preview")
async def generate_preview(request: CharacterPreviewRequest = None):
    """Generate a preview image of the character"""
    try:
        if not app_state.source_face:
            return JSONResponse(
                content={"success": False, "message": "No character face uploaded"},
                status_code=400
            )
        
        # Use provided prompt and seed or generate defaults
        base_prompt = request.base_prompt if request and request.base_prompt else f"portrait of {app_state.physical_description}, {app_state.initial_attire}"
        seed = request.seed if request and request.seed else int(time.time())
        
        print(f"Generating character preview with prompt: {base_prompt}")
        print(f"Using seed: {seed}")
        
        # Generate the image
        image_path, image_message = generate_image_with_face_swap(base_prompt, seed)
        
        if not image_path:
            return JSONResponse(
                content={"success": False, "message": image_message},
                status_code=400
            )
        
        # Store the base prompt and seed for future use
        app_state.character_base_prompt = base_prompt
        app_state.character_seed = seed
        
        # Read and encode the image as base64
        full_path = os.path.join(OUTPUT_DIR, image_path)
        if not os.path.exists(full_path):
            return JSONResponse(
                content={"success": False, "message": f"Generated image not found at {full_path}"},
                status_code=400
            )
            
        with open(full_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return JSONResponse(
            content={
                "success": True,
                "message": "Character preview generated successfully",
                "image_data": image_data,
                "base_prompt": base_prompt,
                "seed": seed
            },
            status_code=200
        )
    except Exception as e:
        print(f"Error generating character preview: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            content={"success": False, "message": f"Error: {str(e)}"},
            status_code=500
        )


@app.post("/regenerate_character_preview")
async def regenerate_preview():
    """Regenerate the character preview with a new seed"""
    try:
        if not app_state.character_base_prompt:
            return JSONResponse(
                content={"success": False, "message": "No character base prompt available"},
                status_code=400
            )
            
        # Generate a new seed
        new_seed = int(time.time())
        
        # Generate the preview image with the new seed
        image_path, image_message = generate_image_with_face_swap(app_state.character_base_prompt, new_seed)
        
        if not image_path:
            return JSONResponse(
                content={"success": False, "message": image_message},
                status_code=400
            )
            
        # Update the character seed
        app_state.character_seed = new_seed
            
        # Read and encode the image as base64
        with open(os.path.join(OUTPUT_DIR, image_path), "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return JSONResponse(
            content={
                "success": True,
                "message": "Character preview regenerated successfully",
                "image_data": image_data,
                "seed": new_seed
            },
            status_code=200
        )
    except Exception as e:
        print(f"Error regenerating character preview: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            content={"success": False, "message": f"Error: {str(e)}"},
            status_code=500
        )


@app.get("/scan_models")
async def scan_models():
    """Scan for available Stable Diffusion models"""
    try:
        models = app_state.scan_available_models()
        return JSONResponse(
            content={"success": True, "models": models},
            status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "message": f"Error scanning models: {str(e)}"},
            status_code=500
        )


class ChatMessage(BaseModel):
    message: str


@app.post("/chat")
async def chat(chat_message: ChatMessage):
    """Process a chat message and return a response with automatic image generation"""
    message = chat_message.message

    # Check if API key is set
    if not app_state.mistral_api_key:
        return JSONResponse(
            content={
                "response": "Please set your Mistral API key in the Setup tab first.",
                "image_data": None,
                "image_message": None
            },
            status_code=200
        )

    # Store the user message
    current_exchange = {"user": message}
    app_state.chat_history.append(current_exchange)

    # Generate a response. If the message requests an image, add a brief hint so the LLM returns IMAGE_PROMPT.
    msg_for_mistral = ("[GENERATE_IMAGE] " + message) if message_requests_image(message) else message
    response_data = generate_mistral_response(msg_for_mistral)
    chat_response = response_data["chat_response"]
    image_prompt = response_data["image_prompt"]

    # Store the assistant's response
    current_exchange["assistant"] = chat_response

    # Check if we should generate an image
    image_data = None
    image_message = None

    
    # Check if this is the first message (excluding system initialization)
    is_first_chat = len(app_state.chat_history) <= 1
    # Clean image_prompt for checking
    cleaned_image_prompt = image_prompt.strip() if image_prompt else ""
    cleaned_image_prompt = re.sub(r'^\*+\s*', '', cleaned_image_prompt)  # Remove leading asterisks
    cleaned_image_prompt = re.sub(r'\s*\*+$', '', cleaned_image_prompt)  # Remove trailing asterisks
    cleaned_image_prompt = cleaned_image_prompt.strip().lower()
    
    if is_first_chat and app_state.physical_description and cleaned_image_prompt and cleaned_image_prompt != "none":
        print("Appending physical description to first image prompt")
        image_prompt = f"{app_state.physical_description}, " + image_prompt
    


    if cleaned_image_prompt and cleaned_image_prompt != "none" and (getattr(app_state, 'face_image_path', None) or getattr(app_state, 'source_img', None)):
        # Extract camera angle from user message if present
        camera_angle = extract_camera_angle(message)
        if camera_angle:
            # Add camera angle to the prompt if not already present
            if camera_angle not in image_prompt.lower():
                image_prompt += f", {camera_angle}"
                


        # Generate image using the dedicated image prompt
        image_path, image_message = generate_image_with_face_swap(image_prompt)
        if image_path:
            # Add image to image history
            image_entry = {
                "path": image_path,
                "prompt": image_prompt,
                "timestamp": time.time()
            }
            app_state.image_history.append(image_entry)

            # Construct full path to the image
            full_image_path = os.path.join(OUTPUT_DIR, image_path)
            # Read and encode the image as base64
            with open(full_image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

    return JSONResponse(
        content={
            "response": chat_response,
            "image_data": image_data,
            "image_message": image_message,
            "image_history": app_state.image_history[-10:] if hasattr(app_state, 'image_history') else []
        },
        status_code=200
    )



@app.post("/clear_chat", response_model=dict)
async def clear_chat():
    """Clear the chat history"""
    app_state.chat_history = []
    return JSONResponse(
        content={"success": True, "message": "Chat history cleared"},
        status_code=200
    )


# =====================
# Gradio UI
# =====================

def create_ui(launch: bool = True):
    """Create the Gradio UI for the chatbot with two-tab layout"""
    with gr.Blocks(title="Fantasy AI Roleplay Chatbot") as demo:
        gr.Markdown("# 🤖 Fantasy AI Roleplay Chatbot")
        gr.Markdown(
            "Upload a character face, set their description, and chat with your AI character who can generate images of themselves!")

        # Create tabs for Setup and Chat
        with gr.Tabs() as tabs:
            # Tab 1: Setup + Character Creator
            with gr.TabItem("Setup & Character Creator"):
                with gr.Row():
                    # Left Column (1/3 width) - API & Model Settings
                    with gr.Column(scale=1):
                        gr.Markdown("## API & Model Settings")
                        
                        # API keys are hard-coded in the backend; UI inputs removed
                        status_box = gr.Textbox(label="Status", interactive=False)
                    
                    # Right Column (2/3 width) - Character Setup
                    with gr.Column(scale=2):
                        gr.Markdown("## Character Setup")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                face_upload = gr.Image(
                                    label="Upload Character Face Image",
                                    type="filepath"
                                )
                                
                                character_name = gr.Textbox(label="Character Name")
                                relation_to_user = gr.Textbox(label="Relation to User")
                                user_name = gr.Textbox(label="What Character Calls User")
                                
                                # New fields
                                gender = gr.Dropdown(
                                    label="Gender",
                                    choices=["Female", "Male", "Ambiguous"],
                                    value="Female"
                                )
                                
                                style = gr.Dropdown(
                                    label="Visual Style",
                                    choices=["Photorealistic", "Cartoon", "3D", "Anime", "Oil Painting"],
                                    value="Photorealistic"
                                )
                            
                            with gr.Column(scale=1):
                                behavioral_desc = gr.Textbox(label="Personality/Behavior", lines=3)
                                physical_desc = gr.Textbox(label="Physical Description", lines=3)
                                initial_attire = gr.Textbox(label="Initial Attire/Clothing", lines=2)
                                chat_context = gr.Textbox(label="Chat Context/Setting", lines=2)
                        
                        # Character Preview removed — ImageRouter will be used for generation on demand
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Placeholder where previews used to be
                                gr.Markdown("Character preview generation removed. Use Chat tab to generate images.")
                            with gr.Column(scale=1):
                                save_character_btn = gr.Button("Save Character", variant="primary")
                        
                        # Function to handle model selection from dropdown
                        def handle_model_selection(model_name):
                            if model_name:
                                return os.path.join(MODELS_DIR, model_name)
                            return ""
                        
                        # API inputs removed; credentials are hard-coded in the backend
                        
                        # (Old remote save_character removed)

                                                # Function to generate character preview
                        def generate_character_preview():
                            try:
                                # Check if face image is uploaded
                                if face_upload.value is None:
                                    print("Face upload value is None in generate_character_preview")
                                    return None, "Please upload a character face image first"
                                
                                # Debug the face upload value
                                print(f"Face upload value in generate_character_preview: {face_upload.value}")
                                
                                # Ensure we have a valid file path
                                if not os.path.exists(face_upload.value):
                                    print(f"Warning: Face image path does not exist: {face_upload.value}")
                                    return None, "Please upload a valid character face image"
                                
                                # Update app_state with the face image path
                                app_state.face_image_path = face_upload.value
                                
                                print(f"Face upload value: {face_upload.value}")  # Debug print
                                
                                # Set the face image path in app_state so the standalone function can use it
                                app_state.face_image_path = face_upload.value
                                print(f"[DEBUG] Setting physical_description in Gradio UI #1: '{physical_desc.value}'")
                                app_state.physical_description = physical_desc.value
                                app_state.initial_attire = initial_attire.value
                                app_state.gender = gender.value
                                app_state.style = style.value
                                
                                # Read uploaded image into memory for later Face++ merging
                                # (no local face detection).
                                _, source_img = extract_face_from_image(app_state.face_image_path)
                                if source_img is None:
                                    return None, "Could not read the uploaded image."

                                app_state.source_face = None
                                app_state.source_img = source_img
                                
                                # Construct preview prompt
                                preview_prompt = f"{app_state.gender}, standing, {app_state.physical_description}, wearing {app_state.initial_attire}, {app_state.style}, high quality, intricate details"
                                app_state.character_base_prompt = preview_prompt
                                
                                # Generate a random seed
                                import random
                                seed = random.randint(100000, 999999)
                                app_state.character_seed = seed
                                
                                # Generate image with face swap using the preview prompt
                                image_path, msg = generate_image_with_face_swap(preview_prompt, seed=seed)
                                if image_path:
                                    return os.path.join(OUTPUT_DIR, os.path.basename(image_path)), msg
                                return None, msg
                            except Exception as e:
                                import traceback
                                print(traceback.format_exc())
                                return None, f"Error generating preview: {str(e)}"
                        
                        # Removed regenerate_character_preview function as it's no longer needed
                        # User will use generate_character_preview repeatedly until satisfied
                        
                        # Function to save character details locally (no HTTP call)
                        def save_character(face_path, name, relation, username, behavioral_desc, physical_desc, 
                                          context, attire, gender, style):
                            try:
                                if not face_path:
                                    return "Please upload a character face image"

                                # Copy uploaded file into UPLOAD_DIR
                                dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{os.path.basename(face_path)}")
                                shutil.copy(face_path, dest_path)
                                # Read the uploaded image (no face detection)
                                _, source_img = extract_face_from_image(dest_path)
                                if source_img is None:
                                    return "Could not read the uploaded image"

                                # Update app_state with the uploaded image path and numpy image
                                app_state.source_face = None
                                app_state.source_img = source_img
                                app_state.face_image_path = dest_path
                                print(f"[DEBUG] Setting physical_description in save_character: '{physical_desc}'")
                                app_state.physical_description = physical_desc
                                app_state.behavioral_description = behavioral_desc
                                app_state.character_name = name
                                app_state.relation_to_user = relation
                                app_state.user_name = username
                                app_state.chat_context = context
                                app_state.initial_attire = attire
                                app_state.gender = gender
                                app_state.style = style

                                return "Character saved locally"
                            except Exception as e:
                                return f"Error: {str(e)}"
                        
                        # Connect save_character to local handler (status_box is defined earlier)
                        save_character_btn.click(
                            save_character,
                            inputs=[face_upload, character_name, relation_to_user, user_name, 
                                   behavioral_desc, physical_desc, chat_context, initial_attire, gender, style],
                            outputs=[status_box]
                        )
                        
                        # Preview generation removed; no click handler
                        
                        # Removed regenerate preview button click handler
                        # User will use generate_character_preview repeatedly until satisfied
            
            # Tab 2: Chat Interface
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            # type="messages"
                        )

                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Type your message here...",
                            lines=2
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear Chat")

                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="Current Generated Image",
                            type="filepath"
                        )
                        image_caption = gr.Markdown("Image will appear here when generated")
                        with gr.Row():
                            regenerate_btn = gr.Button("🔄 Regenerate Image", variant="secondary")

                # Add image gallery
                with gr.Row():
                    gr.Markdown("## Image Gallery")

                with gr.Row():
                    image_gallery = gr.Gallery(
                        label="Previously Generated Images",
                        show_label=True,
                        columns=4,
                        height=200,
                        object_fit="contain"
                    )

                # Helper: normalize various chat history shapes to a list of message dicts
                def _normalize_chat_history(chat_history: list) -> list:
                    """Normalize various chat history shapes to a list of message dicts.

                    Accepts:
                    - list of dicts like [{'role': 'user', 'content': '...'}, ...]
                    - list of (user, assistant) pairs as produced by gr.Chatbot: [('hi','hello'), ...]
                    - empty or None
                    Returns a list of dicts with 'role' and 'content'.
                    """
                    if not chat_history:
                        return []

                    # Already a list of dicts with role keys
                    try:
                        if all(isinstance(m, dict) and 'role' in m for m in chat_history):
                            return list(chat_history)
                    except Exception:
                        pass

                    # If it's a list of pairs (user, assistant)
                    is_pair_list = True
                    for it in chat_history:
                        if not (isinstance(it, (list, tuple)) and len(it) >= 2):
                            is_pair_list = False
                            break

                    if is_pair_list:
                        out = []
                        for user_part, assistant_part in chat_history:
                            out.append({"role": "user", "content": user_part})
                            if assistant_part:
                                out.append({"role": "assistant", "content": assistant_part})
                        return out

                    # Fallback: return as list (best-effort)
                    return list(chat_history)

                # Helper: convert flat messages list (dicts with role/content) to Chatbot pairs list.
                def messages_to_pairs(messages: list) -> list:
                    """Convert flat messages list (dicts with role/content) to Chatbot pairs list.

                    Returns list of (user, assistant) tuples for gr.Chatbot.
                    """
                    pairs = []
                    i = 0
                    while i < len(messages):
                        if messages[i].get("role") == "user":
                            user = messages[i].get("content", "")
                            assistant = ""
                            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                                assistant = messages[i + 1].get("content", "")
                                i += 2
                            else:
                                i += 1
                            pairs.append((user, assistant))
                        else:
                            i += 1
                    return pairs

                def send_message(message, history):
                    # Initialize history as a list if it's None (Gradio default)
                    if history is None:
                        history = []

                    # Normalize incoming history to list of message dicts
                    normalized = _normalize_chat_history(history)

                    # Handle empty messages
                    if not message:
                        return history, "", None, "No message sent", []

                    try:
                        # Call local Mistral wrapper directly (no HTTP needed)
                        resp = generate_mistral_response(message)
                        response_text = resp.get("chat_response", "No response received")
                        image_prompt = resp.get("image_prompt", "none")

                        # Append to normalized history as role/content dicts
                        normalized.append({"role": "user", "content": message})
                        normalized.append({"role": "assistant", "content": response_text})

                        image_display = None
                        image_message = None

                        # Clean and check image_prompt for markdown formatting
                        cleaned_gradio_prompt = image_prompt.strip() if image_prompt else ""
                        cleaned_gradio_prompt = re.sub(r'^\*+\s*', '', cleaned_gradio_prompt)  # Remove leading asterisks
                        cleaned_gradio_prompt = re.sub(r'\s*\*+$', '', cleaned_gradio_prompt)  # Remove trailing asterisks
                        cleaned_gradio_prompt = cleaned_gradio_prompt.strip().lower()
                        
                        # Decide whether to generate an image
                        if cleaned_gradio_prompt and cleaned_gradio_prompt != "none" and (getattr(app_state, 'face_image_path', None) or getattr(app_state, 'source_img', None)):
                            img_path, img_msg = generate_image_with_face_swap(image_prompt)
                            image_message = img_msg
                            if img_path:
                                full_img_path = os.path.join(OUTPUT_DIR, img_path)
                                if os.path.exists(full_img_path):
                                    with open(full_img_path, "rb") as f:
                                        image_bytes = f.read()
                                    image_display = Image.open(BytesIO(image_bytes))

                        # Prepare image gallery items from app_state
                        gallery_items = []
                        for img_entry in getattr(app_state, 'image_history', [])[-10:]:
                            p = img_entry.get('path')
                            if p and os.path.exists(os.path.join(OUTPUT_DIR, p)):
                                gallery_items.append(os.path.join(OUTPUT_DIR, p))

                        # Convert flat message dicts back to gr.Chatbot pairs
                        pairs = messages_to_pairs(normalized)
                        return pairs, "", image_display, image_message or "No image generated", gallery_items

                    except Exception as e:
                        # On error, append an assistant error message and return pairs
                        normalized.append({"role": "user", "content": message})
                        normalized.append({"role": "assistant", "content": f"Error: {str(e)}"})
                        pairs = messages_to_pairs(normalized)
                        return pairs, "", None, f"Error: {str(e)}", []

                def clear_chat_history():
                    try:
                        requests.post("http://127.0.0.1:8000/clear_chat")
                        return [], None, "Chat cleared", []
                    except Exception as e:
                        return [], None, f"Error clearing chat: {str(e)}", []
                        
                                
                def regenerate_current_image():
                    """Regenerate the current image with a new seed"""
                    try:
                        # Generate a new random seed
                        seed = int(time.time())
                        
                        # Call the regenerate_image function
                        image_path, image_message = regenerate_image(seed)
                        
                        if not image_path:
                            return None, image_message
                            
                        # Read and encode the image as base64
                        full_path = os.path.join(OUTPUT_DIR, image_path)
                        with open(full_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            
                        # Get updated image gallery (store but don't return)
                        gallery_items = []
                        for img_entry in app_state.image_history[-10:]:
                            if os.path.exists(os.path.join(OUTPUT_DIR, img_entry["path"])):
                                gallery_items.append(os.path.join(OUTPUT_DIR, img_entry["path"]))
                                
                        # Only return the image and message, not the gallery items
                        return Image.open(full_path), "Image regenerated with new seed"
                    except Exception as e:
                        print(f"Error regenerating image: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        return None, f"Error: {str(e)}", []

                # Connect UI components to functions
                submit_btn.click(
                    send_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg, image_output, image_caption, image_gallery]
                )

                msg.submit(
                    send_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg, image_output, image_caption, image_gallery]
                )

                clear_btn.click(
                    clear_chat_history,
                    outputs=[chatbot, image_output, image_caption, image_gallery]
                )
                
                regenerate_btn.click(
                    regenerate_current_image,
                    outputs=[image_output, image_caption]
                )

        # Help tab
        with gr.Tab("Help"):
            gr.Markdown("""
            ## Fantasy AI Roleplay Chatbot Help

            ### Required Setup
            1. **Mistral API Key**: Required for generating text responses
            2. **Face Image**: Upload a clear face image for your character
            3. **Character Description**: Describe your character's physical appearance

            ### Optional Setup
            - **Stable Diffusion Model**: Select from available models or provide a custom path
              (If not provided, image generation will be disabled)

            ### Character Creation Tips
            - For face swapping, use a clear frontal face image with good lighting
            - Be detailed in your character description (hair color, eye color, etc.)
            - Specify the initial attire/clothing for consistent images
            - Select the appropriate gender and visual style for better results

            ### Chat Tips
            - Visual triggers in conversation will generate images automatically
            - Image generation works best with concrete visual descriptions
            - You can regenerate images with a new seed if you don't like the result

            ### Troubleshooting
            - If face swapping fails, try a different source image with a clearer face
            - CUDA is required for optimal performance
            - Make sure your Mistral API key is valid
            """)

    # At this point `demo` is the Gradio Blocks instance built in the `with` block.
    # If launch=True, build a single ASGI app that mounts FastAPI and Gradio and
    # run uvicorn. If launch=False, return the Blocks instance so the module can
    # mount it and expose `root_app` for external servers (e.g., Heroku's uvicorn).
    import uvicorn
    from fastapi import FastAPI as _FastAPI

    if launch:
        gradio_app = demo.get_app()
        root_app = _FastAPI()
        root_app.mount("/api", app)
        root_app.mount("/", gradio_app)

        # Use PORT env var if provided (Heroku) otherwise default to 7860
        port = int(os.environ.get("PORT", 7860))
        uvicorn.run(root_app, host="0.0.0.0", port=port)
    else:
        return demo


# When imported by an ASGI server (uvicorn/gunicorn) we should expose `root_app`.
# Build the Gradio demo without launching it and mount it together with the API.
try:
    demo = create_ui(launch=False)
    from fastapi import FastAPI as _FastAPI

    # Some Gradio versions expose an ASGI app via demo.get_app(), newer or
    # older versions may instead provide a mount helper. Try both to be
    # compatible across versions.
    gradio_app = None
    try:
        gradio_app = demo.get_app()
    except AttributeError:
        import gradio as gr
        if hasattr(gr, "mount_gradio_app"):
            # We'll call mount_gradio_app later after creating root_app
            gr_mount_helper = gr.mount_gradio_app
        else:
            raise

    root_app = _FastAPI()
    root_app.mount("/api", app)
    # Mount Gradio at /gradio to avoid conflicts and make the UI reachable
    if gradio_app is not None:
        root_app.mount("/gradio", gradio_app)
    else:
        # Use gradio's helper to mount the Blocks instance into the FastAPI app
        try:
            gr_mount_helper(root_app, demo, path="/gradio")
        except Exception:
            # Re-raise to be caught by outer exception handler so we log it
            raise

    # Provide a simple redirect from root to the Gradio UI
    from fastapi.responses import RedirectResponse

    @root_app.get("/")
    async def _root_redirect():
        return RedirectResponse(url="/gradio")
except Exception as e:
    # If building the demo fails during import, record the traceback to help
    # debugging and expose a helpful fallback endpoint instead of silently
    # returning 404 for '/'. This makes the error visible when uvicorn starts.
    import traceback
    tb = traceback.format_exc()
    try:
        with open("gradio_init_error.log", "w", encoding="utf-8") as f:
            f.write(tb)
    except Exception:
        # ignore file write errors
        pass
    print("Failed to initialize Gradio demo during import. See gradio_init_error.log for details.")
    print(tb)

    # Expose the API-only app but provide a /gradio fallback that shows the error
    from fastapi.responses import PlainTextResponse
    root_app = app

    @root_app.get("/gradio")
    async def _gradio_unavailable():
        message = (
            "Gradio UI failed to initialize on import.\n"
            "Check gradio_init_error.log in the app directory for the full traceback.\n"
            "If you're running inside a restricted environment, ensure all requirements (gradio, starlette, fastapi) are installed.\n"
        )
        return PlainTextResponse(message, status_code=500)


if __name__ == "__main__":
    # Run the combined ASGI app directly (useful for local testing)
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(root_app, host="0.0.0.0", port=port)