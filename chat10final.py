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
import torch
import insightface

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from io import BytesIO
from PIL import Image

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

# Set up directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
MODELS_DIR = "models"
for directory in [UPLOAD_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize FastAPI
app = FastAPI()
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# Initialize face models at startup
print("Pre-loading face models at startup...")
face_app = insightface.app.FaceAnalysis(
    name="buffalo_l", 
    root="models",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load face swapper
face_swapper = None
try:
    model_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
    if os.path.exists(model_path):
        face_swapper = insightface.model_zoo.get_model(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print("Face swapper loaded successfully")
    else:
        print(f"Warning: Face swapper model not found at {model_path}")
except Exception as e:
    print(f"Warning: Could not load face swapper: {str(e)}")


# Global settings and state
class AppState:
    def __init__(self):
        self.mistral_api_key = ""
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
        # Face swap components - use the pre-loaded models
        self.face_app = face_app
        self.face_swapper = face_swapper
        self.source_face = None
        self.source_img = None
        self.sd_pipe = None
        self.prompt_cache = {}  # Cache for prompt->image mapping
        self.force_clothed = True  # Default: keep character fully clothed
        self.last_used_prompt = None
        # New character attributes
        self.initial_attire = ""
        self.gender = "Female"  # Default gender
        self.style = "Photorealistic"  # Default style
        self.character_base_prompt = ""
        self.character_seed = None
        self.available_models = []  # Will store available SD models

    def apply_nsfw_filter(self, prompt_text, character_desc=""):
        """
        Centralized function to apply NSFW filtering to prompts
        Returns the filtered prompt and a boolean indicating if filtering was applied
        """
        clothing_terms = ["clothed", "jacket", "robe", "uniform", "dress", "outfit", "shirt", "pants", "skirt", "blouse"]
        nsfw_terms = {
            "breast": "chest", 
            "breasts": "chest",
            "boob": "chest", 
            "boobs": "chest",
            "saggy": "natural",
            "areola": "chest",
            "nipple": "chest",
            "nipples": "chest",
            "ass": "figure", 
            "butt": "figure", 
            "naked": "clothed", 
            "nude": "clothed", 
            "revealing": "modest", 
            "cleavage": "neckline", 
            "underwear": "clothing", 
            "lingerie": "clothing",
            "busty": "curvy",
            "voluptuous": "curvy"
        }
        
        # Check if filtering is enabled - if not, return original prompt
        if not self.force_clothed:
            print("NSFW filter is disabled - not applying content filtering")
            return prompt_text, False
            
        needs_clothing = True  # Since force_clothed is True, we need clothing
        filtering_applied = False

        # NSFW detected in physical description
        if character_desc and any(term in character_desc.lower() for term in nsfw_terms):
            needs_clothing = True
        
        # Apply regex pattern matching for compound NSFW phrases
        nsfw_patterns = [
            (r'\b(huge|large|big|saggy|perky)\s+(breast|breasts|boob|boobs)\b', 'curvy figure'),
            (r'\b(dark|pink|large)\s+(areola|nipple|nipples)\b', 'chest'),
            (r'\b(revealing|low-cut|tight)\s+(outfit|dress|top|clothing)\b', 'modest outfit'),
            (r'\b(sexy|seductive|provocative)\s+(pose|look|appearance)\b', 'natural pose')
        ]
        
        # First apply word-by-word replacement
        prompt_lower = prompt_text.lower()
        for nsfw_term, replacement in nsfw_terms.items():
            # Only replace if the term exists and isn't part of another word
            if nsfw_term in prompt_lower:
                # Use regex to replace the term as a whole word
                prompt_text = re.sub(r'\b' + nsfw_term + r'\b', replacement, prompt_text, flags=re.IGNORECASE)
                filtering_applied = True
        
        # Then apply pattern matching for compound phrases
        for pattern, replacement in nsfw_patterns:
            if re.search(pattern, prompt_text, re.IGNORECASE):
                prompt_text = re.sub(pattern, replacement, prompt_text, flags=re.IGNORECASE)
                filtering_applied = True
        
        # Only add generic clothing tag if no specific clothing or colors are already present
        color_keywords = ["red", "blue", "green", "yellow", "orange", "pink", "purple", "brown", "black", "white", "gray", "grey", "teal", "cyan"]
        has_color = any(c in prompt_text.lower() for c in color_keywords)
        has_clothing = any(term in prompt_text.lower() for term in clothing_terms)

        if not has_color and not has_clothing:
            prompt_text = "fully clothed, wearing modest outfit, " + prompt_text
            print("Applied clothing protection to prompt")
            filtering_applied = True
        else:
            print("Color or clothing terms already present — skipping generic clothing prepend")

        
        return prompt_text, filtering_applied

    def initialize_face_models(self):
        """Initialize face detection and swapping models with optimized loading"""
        # Check if models are already loaded
        if hasattr(self, 'face_app') and self.face_app and hasattr(self, 'face_swapper') and self.face_swapper:
            print("Face models already loaded, skipping initialization")
            return True
            
        try:
            print("Initializing face models (this may take a moment the first time)...")
            import time
            start_time = time.time()
            
            # Load face analyzer with optimized settings
            print("Loading face detection model...")
            self.face_app = insightface.app.FaceAnalysis(
                name="buffalo_l", 
                root="models",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition'] # Only load necessary modules
            )
            self.face_app.prepare(ctx_id=0, det_size=(320, 320)) # Use smaller detection size for faster loading
            
            # Load face swapper with optimized settings
            print("Loading face swapping model...")
            self.face_swapper = insightface.model_zoo.get_model(
                'models/inswapper_128.onnx',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            elapsed_time = time.time() - start_time
            print(f"Face models loaded successfully in {elapsed_time:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error initializing face models: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def initialize_sd_model(self):
        """Initialize the Stable Diffusion model with the specified settings"""
        try:
            if not self.sd_model_path:
                print("No SD model path provided")
                return False

            print(f"Loading Stable Diffusion model from {self.sd_model_path}")
            
            # Create the appropriate scheduler based on user selection
            if self.scheduler_type == "dpm_2m_karras":
                scheduler = DPMSolverMultistepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000,
                    solver_order=2,
                    prediction_type="epsilon",
                    thresholding=False,
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint"
                )
            elif self.scheduler_type == "euler_a":
                scheduler = EulerAncestralDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000
                )
            else:
                # Default to DPM++ 2M Karras if unknown
                scheduler = DPMSolverMultistepScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    num_train_timesteps=1000,
                    solver_order=2,
                    prediction_type="epsilon",
                    thresholding=False,
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint"
                )
                
            # Load the pipeline with the selected scheduler
            # Use from_single_file for .safetensors files
            if self.sd_model_path.endswith('.safetensors'):
                self.sd_pipe = StableDiffusionPipeline.from_single_file(
                    self.sd_model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None
                )
            else:
                # Use from_pretrained for model folders
                self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                    self.sd_model_path,
                    scheduler=scheduler,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None
                )
            
            # Apply CLIP skip setting
            if hasattr(self.sd_pipe, 'text_encoder') and self.clip_skip > 1:
                self.sd_pipe._text_encoder = lambda *args, **kwargs: self.sd_pipe.text_encoder(
                    *args, **kwargs
                )[0][-self.clip_skip]
                
            # Move to GPU if available
            if torch.cuda.is_available():
                self.sd_pipe = self.sd_pipe.to("cuda")
                # Enable memory efficient attention if available
                try:
                    import xformers
                    if hasattr(self.sd_pipe, "enable_xformers_memory_efficient_attention"):
                        self.sd_pipe.enable_xformers_memory_efficient_attention()
                        print("Successfully enabled xformers memory efficient attention")
                except ImportError:
                    print("xformers not available, using default attention mechanism")
                except Exception as e:
                    print(f"Could not enable memory efficient attention: {str(e)}")
            
            return True
        except Exception as e:
            print(f"Error initializing SD model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def scan_available_models(self):
        """Scan the models directory for available Stable Diffusion models"""
        model_extensions = ['.safetensors', '.ckpt', '.pth']
        models = []
        
        # Scan for models in the models directory
        for ext in model_extensions:
            models.extend(glob.glob(os.path.join(MODELS_DIR, f"*{ext}")))
        
        # Format model names for display
        self.available_models = [os.path.basename(model) for model in models]
        return self.available_models

    def generate_character_preview(self):
        """Generate a preview image of the character based on current settings"""
        if not self.source_face or not self.source_img:
            return None, "Please upload a character face image first"
            
        # Create a prompt for the character preview
        preview_prompt = f"{self.gender}, standing, {self.physical_description}, wearing {self.initial_attire}, {self.style}, high quality, intricate details"
        
        # Generate the image
        image_path, image_message = generate_image_with_face_swap(preview_prompt)
        
        # Save the base prompt and seed for future use
        if image_path:
            self.character_base_prompt = preview_prompt
            # Store the seed if available
            if hasattr(self, 'last_used_seed') and self.last_used_seed:
                self.character_seed = self.last_used_seed
                
        return image_path, image_message


app_state = AppState()


# =====================
# Face Swap Implementation
# =====================

def extract_face_from_image(image_path):
    """Extract a face from an image with optimized performance"""
    try:
        # Use a lightweight face detector for the initial check
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None, None
            
        # Convert to RGB for insightface
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use the pre-loaded face_app that has both detection and recognition
        if not app_state.face_app:
            print("Face app not initialized, initializing now...")
            # Make sure to include recognition module
            face_app = insightface.app.FaceAnalysis(
                name="buffalo_l", 
                root="models",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']  # Include recognition
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            app_state.face_app = face_app
        
        # Detect faces with full processing (including embeddings)
        faces = app_state.face_app.get(img_rgb)
        
        if not faces:
            print("No face detected in the image")
            return None, None
        
        # Verify that the face has the normed_embedding attribute
        if not hasattr(faces[0], 'normed_embedding') or faces[0].normed_embedding is None:
            print("Warning: Face detected but no embedding generated. Reinitializing face app with recognition...")
            # Reinitialize with recognition explicitly
            app_state.face_app = insightface.app.FaceAnalysis(
                name="buffalo_l", 
                root="models",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition']
            )
            app_state.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Try again with the reinitialized face app
            faces = app_state.face_app.get(img_rgb)
            
            if not faces or not hasattr(faces[0], 'normed_embedding') or faces[0].normed_embedding is None:
                print("Failed to generate face embedding even after reinitialization")
                return None, None
        
        # Use the first detected face
        return faces[0], img_rgb
    except Exception as e:
        print(f"Error extracting face: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None


def swap_face(source_face, source_img, target_path, output_path):
    """Swap face from source to target"""
    try:
        # Check if we have necessary components
        if not app_state.face_swapper or not app_state.face_app:
            raise ValueError("Face models not initialized")

        # Read target image
        target_img = cv2.imread(target_path)
        if target_img is None:
            raise ValueError(f"Could not read target image: {target_path}")

        # Convert to RGB for insightface
        target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Progressive face detection strategy
        # Try with current detection settings first
        target_faces = app_state.face_app.get(target_img_rgb)
        
        # If no faces detected, try with different detection sizes without reconfiguring the model
        if not target_faces:
            print("No faces detected with default settings, trying with alternative detection approach")
            
            # Option 1: Try with image scaling instead of reconfiguring the model
            scale_factors = [0.5, 1.5, 0.75]  # Try different scales
            for scale in scale_factors:
                h, w = target_img_rgb.shape[:2]
                scaled_img = cv2.resize(target_img_rgb, (int(w*scale), int(h*scale)))
                scaled_faces = app_state.face_app.get(scaled_img)
                
                if scaled_faces:
                    print(f"Face detected at scale factor {scale}")
                    # Adjust face coordinates back to original scale
                    for face in scaled_faces:
                        face.bbox = [coord/scale for coord in face.bbox]
                        face.landmark_2d_106 = face.landmark_2d_106/scale if hasattr(face, 'landmark_2d_106') else None
                        face.landmark_3d_68 = face.landmark_3d_68/scale if hasattr(face, 'landmark_3d_68') else None
                    target_faces = scaled_faces
                    break
        
        if not target_faces:
            print("No faces detected even with alternative approaches")
            # Save original image as output
            cv2.imwrite(output_path, target_img)
            return output_path

        # Determine target gender from character description
        target_gender = app_state.gender.lower() if hasattr(app_state, 'gender') else detect_gender_from_description(app_state.physical_description)
        print(f"Target gender from description: {target_gender}")
        
        # Process each face in the target image
        result_img = target_img_rgb.copy()
        faces_swapped = 0
        
        for target_face in target_faces:
            # Detect gender of the face in the generated image
            face_gender = detect_face_gender(target_face)
            print(f"Detected face gender: {face_gender}")
            
            # Only swap if gender matches or if we couldn't determine gender
            if target_gender == "unknown" or face_gender == "unknown" or target_gender == face_gender:
                result_img = app_state.face_swapper.get(result_img, target_face, source_face, paste_back=True)
                faces_swapped += 1
            else:
                print(f"Skipping face swap due to gender mismatch (target: {target_gender}, face: {face_gender})")

        print(f"Swapped {faces_swapped} out of {len(target_faces)} faces based on gender matching")
        
        # Convert back to BGR for OpenCV and save
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_img_bgr)
        print(f"Successfully saved face-swapped image to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error swapping face: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # In case of error, return the original image
        try:
            shutil.copy(target_path, output_path)
            return output_path
        except:
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
        # Check if the face has gender attribute (some insightface models provide this)
        if hasattr(face, 'gender') and face.gender is not None:
            return "female" if face.gender == 0 else "male"
        
        # Fallback to simple heuristics based on face shape
        # These are very rough approximations and not scientifically accurate
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106
            
            # Calculate face width to height ratio
            face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
            face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            ratio = face_width / face_height if face_height > 0 else 0
            
            # Very rough heuristic - women tend to have slightly more oval faces
            # This is NOT accurate and should be replaced with a proper classifier
            if ratio < 0.85:
                return "female"
            elif ratio > 0.95:
                return "male"
        
        # If we can't determine, return unknown
        return "unknown"
    except:
        return "unknown"


# =====================
# Image Generation
# =====================

def generate_image(prompt, seed=None):
    """Generate an image using Stable Diffusion"""
    try:
        if not app_state.sd_pipe:
            return None, "Stable Diffusion model not initialized"

        # Apply NSFW filtering to the prompt
        filtered_prompt, filtering_applied = app_state.apply_nsfw_filter(prompt, app_state.physical_description)
        if filtering_applied:
            print(f"Applied NSFW filtering to prompt:\nOriginal: {prompt}\nFiltered: {filtered_prompt}")
            prompt = filtered_prompt

        # Check if we have this prompt in cache
        if prompt in app_state.prompt_cache and os.path.exists(app_state.prompt_cache[prompt]):
            print(f"Using cached image for prompt: {prompt[:30]}...")
            return app_state.prompt_cache[prompt], "Retrieved from cache"

        # Generate a unique filename
        output_filename = f"generated_{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Set the seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
            print(f"Using seed: {seed}")

        # Generate the image
        print(f"Generating image with prompt: {prompt[:50]}...")
        with torch.inference_mode():
            image = app_state.sd_pipe(
                prompt=prompt,
                guidance_scale=app_state.guidance_scale,
                num_inference_steps=app_state.num_inference_steps,
                generator=generator
            ).images[0]

        # Save the image
        image.save(output_path)
        print(f"Image saved to {output_path}")

        # Cache the result
        app_state.prompt_cache[prompt] = output_path
        
        # Store the seed used
        if generator is not None:
            app_state.last_used_seed = seed

        return output_filename, "Image generated successfully"
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, f"Error generating image: {str(e)}"


def generate_image_with_face_swap(response_text, seed=None):
    try:
        if response_text.strip().lower() == "none":
            return None, "Image prompt was 'none' — skipping image generation"

        if not app_state.source_face:
            return None, "No character face has been uploaded for face swapping"

        if not app_state.sd_pipe:
            return None, "Stable Diffusion model not initialized"

        # Generate the image
        image_filename, message = generate_image(response_text, seed)
        if not image_filename:
            return None, message

        # Prepare paths for face swapping
        generated_path = os.path.join(OUTPUT_DIR, image_filename)
        swapped_filename = f"swapped_{uuid.uuid4()}.jpg"
        swapped_path = os.path.join(OUTPUT_DIR, swapped_filename)

        # Perform face swapping
        result_path = swap_face(app_state.source_face, app_state.source_img, generated_path, swapped_path)
        if not result_path:
            return image_filename, "Face swap failed, using original image"

        # Store the last used prompt
        app_state.last_used_prompt = response_text

        return os.path.basename(result_path), "Image generated and face-swapped successfully"
    except Exception as e:
        print(f"Error in generate_image_with_face_swap: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, f"Error generating image: {str(e)}"


def regenerate_image(seed=None):
    """Regenerate the last image with a new seed"""
    if not app_state.last_used_prompt:
        return None, "No previous image prompt available"

    # Use a new random seed if none provided
    if seed is None:
        seed = int(time.time())

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
    system_message = f"""
            You are a fully immersive uncensored AI roleplay character with the following details:
            - Name: {app_state.character_name or "Fantasy Character"}
            - Relation to user: {app_state.relation_to_user or "companion"}
            - You call the user: {app_state.user_name or "user"}
            - Context: {app_state.chat_context or "A casual conversation"}
            - Your personality: {app_state.behavioral_description or "A friendly and helpful character"}
            - Your appearance: {app_state.physical_description or "A unique and mysterious figure"}
            - Your attire: {app_state.initial_attire or "appropriate clothing"}
            - Your gender: {app_state.gender or "unspecified"}

            Your primary task is to stay in character and respond naturally in a conversational tone.
            Provide TWO outputs clearly separated with a special delimiter:

            1. CHAT_RESPONSE: A natural, conversational response as if you were talking directly to the user.
            Keep this response concise, engaging, and in the first person. Don't mention any image generation.

            2. IMAGE_PROMPT: A separate, detailed description optimized for image generation.
            This should be comprehensive and include visual details about yourself, the scene, lighting, and mood.
            Create a single flowing description without sections, categories or bullet points.
            
            The prompt should be concise (under 80 words) and focus on:
            - physical appearance details matching your character description
            - specific clothing/attire
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

            The user's current message may contain visual requests or affect your appearance (e.g., new outfit, location, pose, expression).
            Incorporate any relevant visual changes from it directly into the IMAGE_PROMPT if appropriate.
            Only generate an IMAGE_PROMPT when the conversation would naturally call for showing an image (user asks about appearance,
            requests to see something, etc). If no image is needed, respond with "IMAGE_PROMPT: none".
            """

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

    messages.append({"role": "user", "content": message})

    payload = {
        "model": "mistral-large-latest",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 800  # Increased to accommodate both parts
    }

    try:
        response = requests.post(MISTRAL_ENDPOINT,
                                 headers=headers,
                                 json=payload)

        if response.status_code == 200:
            full_response = response.json()["choices"][0]["message"]["content"]

            # Parse the response to separate chat response and image prompt
            chat_response = ""
            image_prompt = "none"

            # Extract chat response
            chat_match = re.search(r"CHAT_RESPONSE:\s*(.*?)(?=IMAGE_PROMPT:|$)", full_response, re.DOTALL)
            if chat_match:
                chat_response = chat_match.group(1).strip()

            # Extract image prompt
            prompt_match = re.search(r"IMAGE_PROMPT:\s*(.*?)$", full_response, re.DOTALL)
            if prompt_match:
                image_prompt = prompt_match.group(1).strip()

            # If no proper formatting was used, use the full response as chat_response
            if not chat_response:
                chat_response = full_response

            return {
                "chat_response": chat_response,
                "image_prompt": "none" if image_prompt.lower() == "none" else image_prompt
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
        "here I am", "what I look like", "let me show", "here's what",
        "I appear", "I look", "you'd see"
    ]

    response_lower = response.lower()
    return any(trigger in response_lower for trigger in visual_triggers)

# ✅ Genitalia Enhancement Logic
def apply_explicit_genitalia_enhancement(prompt: str, gender: str, force_clothed: bool) -> str:
    if force_clothed:
        return prompt

    prompt_lower = prompt.lower()
    bottom_off_keywords = [
        "bottomless", "no panties", "panties removed", "panties off",
        "fully nude", "naked from below", "wearing only a top",
        "no skirt", "petticoat removed", "without panties"
    ]
    bottom_off = any(kw in prompt_lower for kw in bottom_off_keywords)

    if bottom_off:
        if gender == "female" and not any(term in prompt_lower for term in ["vagina", "pussy"]):
            prompt += ", visible wet vagina"
        elif gender == "male" and not any(term in prompt_lower for term in ["penis", "cock"]):
            prompt += ", visible erect penis"
        elif gender == "ambiguous" and "genitals" not in prompt_lower:
            prompt += ", exposed genitals"

    return prompt

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
    sd_model_path: Optional[str] = ""
    scheduler_type: Optional[str] = "dpm_2m_karras"
    guidance_scale: Optional[float] = 7.0
    num_inference_steps: Optional[int] = 30
    clip_skip: Optional[int] = 2


@app.post("/set_api_settings")
async def set_api_settings(settings: ApiSettings):
    """Set the API keys and model settings"""
    app_state.mistral_api_key = settings.mistral_api_key
    
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

        # Extract face with embeddings
        source_face, source_img = extract_face_from_image(file_location)
        
        if not source_face or not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
            return JSONResponse(
                content={"success": False, "message": "Failed to extract face embedding from the uploaded image"},
                status_code=400
            )
        
        # Store the face and image path
        app_state.source_face = source_face
        app_state.source_img = source_img
        app_state.face_image_path = file_location
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

        # Create a simple face preview image
        bbox = source_face.bbox.astype(int)
        face_crop = source_img[max(0, bbox[1]):min(source_img.shape[0], bbox[3]), 
                          max(0, bbox[0]):min(source_img.shape[1], bbox[2])]
        
        # Ensure we have a valid crop
        if face_crop.size == 0:
            # Fallback to simple center crop if bbox is invalid
            h, w = source_img.shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            face_crop = source_img[center_y-size:center_y+size, center_x-size:center_x+size]
        
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

    # Toggle clothing protection based on message content and token length
    nsfw_trigger_words = ["strip", "undress", "remove clothes", "take off", "bare", "naked", 
                         "nude", "expose", "lingerie", "underwear", "nsfw", "xxx", "porn",
                         "disable filter", "turn off filter", "no filter"]
    restore_clothing_words = ["clothed", "dress up", "put clothes", "wear something", 
                             "get dressed", "modest", "covered", "enable filter", 
                             "turn on filter", "activate filter"]

    lower_msg = message.lower()
    
    # Smart NSFW logic - check if request implies nudity and token length
    tokens = len(message.split())
    if any(word in lower_msg for word in nsfw_trigger_words) and tokens <= 77:
        app_state.force_clothed = False
        print("NSFW filter disabled by user request (short message with NSFW trigger)")
    elif any(word in lower_msg for word in restore_clothing_words):
        app_state.force_clothed = True
        print("NSFW filter enabled by user request")

    # Generate a response
    response_data = generate_mistral_response(message)
    chat_response = response_data["chat_response"]
    image_prompt = response_data["image_prompt"]

    # Store the assistant's response
    current_exchange["assistant"] = chat_response

    # Check if we should generate an image
    image_data = None
    image_message = None

    if image_prompt != "none" and app_state.source_face is not None:
        # Extract camera angle from user message if present
        camera_angle = extract_camera_angle(message)
        if camera_angle:
            # Add camera angle to the prompt if not already present
            if camera_angle not in image_prompt.lower():
                image_prompt += f", {camera_angle}"
                
        # Apply genitalia enhancement if NSFW filter is disabled
        if not app_state.force_clothed:
            image_prompt = apply_explicit_genitalia_enhancement(image_prompt, app_state.gender.lower(), app_state.force_clothed)

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

def create_ui():
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
                        
                        mistral_api_key = gr.Textbox(
                            label="Mistral API Key",
                            placeholder="Enter your Mistral API key",
                            type="password"
                        )
                        
                        # Scan models folder for available models
                        def get_available_models():
                            return app_state.scan_available_models()
                        
                        sd_model_dropdown = gr.Dropdown(
                            label="Stable Diffusion Model",
                            choices=get_available_models(),
                            value=None,
                            interactive=True,
                            allow_custom_value=True
                        )
                        
                        sd_model_path = gr.Textbox(
                            label="Custom Model Path (Optional)",
                            placeholder="E.g. C:/models/v1-5-pruned-emaonly.safetensors",
                            value=""
                        )
                        
                        gr.Markdown("### Image Generation Settings")
                        
                        scheduler_type = gr.Dropdown(
                            label="Sampler",
                            choices=["dpm_2m_karras", "euler_a"],
                            value="dpm_2m_karras",
                            info="DPM++ 2M Karras or Euler A"
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1.0,
                            maximum=15.0,
                            value=7.0,
                            step=0.1,
                            info="Higher values = stronger adherence to prompt"
                        )
                        
                        num_inference_steps = gr.Slider(
                            label="Steps",
                            minimum=10,
                            maximum=50,
                            value=14,
                            step=1,
                            info="More steps = higher quality but slower"
                        )
                        
                        clip_skip = gr.Slider(
                            label="CLIP Skip",
                            minimum=1,
                            maximum=4,
                            value=2,
                            step=1,
                            info="Higher values = less detail but more creative"
                        )
                        
                        save_api_btn = gr.Button("Save API & Model Settings")
                    
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
                        
                        gr.Markdown("## Character Preview")
                        with gr.Row():
                            with gr.Column(scale=1):
                                preview_image = gr.Image(label="Character Preview", height=400)
                                preview_status = gr.Textbox(label="Preview Status", interactive=False)
                            
                            with gr.Column(scale=1):
                                generate_preview_btn = gr.Button("Generate Character Preview", variant="primary")
                                regenerate_preview_btn = gr.Button("Regenerate Preview", variant="secondary")
                                save_character_btn = gr.Button("Save Character", variant="primary")
                        
                        # Function to handle model selection from dropdown
                        def handle_model_selection(model_name):
                            if model_name:
                                return os.path.join(MODELS_DIR, model_name)
                            return ""
                        
                        # Function to save API and model settings
                        def save_api_settings(api_key, model_path, custom_path, scheduler, guidance, steps, clip):
                            try:
                                # Use custom path if provided, otherwise use selected model
                                final_model_path = custom_path if custom_path else model_path
                                
                                api_response = requests.post(
                                    "http://127.0.0.1:8000/set_api_settings",
                                    json={
                                        "mistral_api_key": api_key, 
                                        "sd_model_path": final_model_path,
                                        "scheduler_type": scheduler,
                                        "guidance_scale": guidance,
                                        "num_inference_steps": steps,
                                        "clip_skip": clip
                                    }
                                )
                                if not api_response.ok:
                                    return api_response.json().get("message", "API settings error")
                                return api_response.json().get("message", "API settings saved successfully!")
                            except Exception as e:
                                return f"Error: {str(e)}"
                        
                        # Function to save character details
                        def save_character(face_path, name, relation, username, behavioral_desc, physical_desc, 
                                          context, attire, gender, style):
                            try:
                                if not face_path:
                                    return "Please upload a character face image"
                                
                                files = {"face_image": open(face_path, "rb")}
                                char_response = requests.post(
                                    "http://127.0.0.1:8000/upload_character",
                                    files=files,
                                    data={
                                        "physical_description": physical_desc,
                                        "behavioral_description": behavioral_desc,
                                        "character_name": name,
                                        "relation_to_user": relation,
                                        "user_name": username,
                                        "chat_context": context,
                                        "initial_attire": attire,
                                        "gender": gender,
                                        "style": style
                                    }
                                )
                                if not char_response.ok:
                                    return char_response.json().get("message", "Character upload failed")
                                return char_response.json().get("message", "Character saved successfully!")
                            except Exception as e:
                                return f"Error: {str(e)}"
                        
                                                # Function to generate character preview
                        def generate_character_preview():
                            try:
                                # Check if face image is uploaded
                                if face_upload.value is None:
                                    return None, "Please upload a character face image first"
                                
                                print(f"Face upload value: {face_upload.value}")  # Debug print
                                
                                # First save the character details to ensure face extraction happens
                                # This calls the /upload_character endpoint which extracts the face using insightface
                                save_result = save_character(
                                    face_upload.value,
                                    character_name.value,
                                    relation_to_user.value,
                                    user_name.value,
                                    behavioral_desc.value,
                                    physical_desc.value,
                                    chat_context.value,
                                    initial_attire.value,
                                    gender.value,
                                    style.value
                                )
                                
                                # Check if there was an error saving the character
                                if "Error" in save_result or "failed" in save_result.lower():
                                    return None, save_result
                                
                                # Generate a base prompt for the character
                                base_prompt = f"{style.value} style, portrait of {physical_desc.value}, {initial_attire.value}, high quality, detailed"
                                
                                # Generate a random seed
                                seed = int(time.time())
                                
                                # Call the backend to generate the preview
                                preview_response = requests.post(
                                    "http://127.0.0.1:8000/generate_character_preview",
                                    json={"base_prompt": base_prompt, "seed": seed}
                                )
                                
                                if not preview_response.ok:
                                    error_msg = preview_response.json().get("message", "Preview generation failed")
                                    print(f"Preview generation error: {error_msg}")
                                    return None, error_msg
                                
                                data = preview_response.json()
                                if data.get("image_data"):
                                    import base64
                                    from PIL import Image
                                    import io
                                    image_bytes = base64.b64decode(data["image_data"])
                                    image = Image.open(io.BytesIO(image_bytes))
                                    return image, f"Preview generated successfully! Seed: {data.get('seed', 'unknown')}"
                                return None, data.get("message", "No preview image generated")
                            except Exception as e:
                                import traceback
                                print(traceback.format_exc())
                                return None, f"Error generating preview: {str(e)}"
                        
                        # Function to regenerate character preview
                        def regenerate_character_preview():
                            try:
                                # Simply call the generate preview function again
                                # This will use a new random seed
                                return generate_character_preview()
                            except Exception as e:
                                return None, f"Error regenerating preview: {str(e)}"
                        
                        # Function to save character details
                        def save_character(face_path, name, relation, username, behavioral_desc, physical_desc, 
                                          context, attire, gender, style):
                            try:
                                if not face_path:
                                    return "Please upload a character face image"
                                
                                files = {"face_image": open(face_path, "rb")}
                                char_response = requests.post(
                                    "http://127.0.0.1:8000/upload_character",
                                    files=files,
                                    data={
                                        "physical_description": physical_desc,
                                        "behavioral_description": behavioral_desc,
                                        "character_name": name,
                                        "relation_to_user": relation,
                                        "user_name": username,
                                        "chat_context": context,
                                        "initial_attire": attire,
                                        "gender": gender,
                                        "style": style
                                    }
                                )
                                
                                if not char_response.ok:
                                    return char_response.json().get("message", "Character upload failed")
                                
                                # Save the current base prompt and seed if available
                                if hasattr(app_state, 'character_base_prompt') and app_state.character_base_prompt:
                                    # This will be handled by the backend automatically when character is saved
                                    return "Character saved successfully with base prompt and seed!"
                                
                                return char_response.json().get("message", "Character saved successfully!")
                            except Exception as e:
                                return f"Error: {str(e)}"
                        
                        # Connect UI components to functions
                        sd_model_dropdown.change(handle_model_selection, inputs=[sd_model_dropdown], outputs=[sd_model_path])
                        
                        save_api_btn.click(
                            save_api_settings,
                            inputs=[mistral_api_key, sd_model_path, sd_model_path, scheduler_type, 
                                   guidance_scale, num_inference_steps, clip_skip],
                            outputs=[preview_status]
                        )
                        
                        save_character_btn.click(
                            save_character,
                            inputs=[face_upload, character_name, relation_to_user, user_name, 
                                   behavioral_desc, physical_desc, chat_context, initial_attire, gender, style],
                            outputs=[preview_status]
                        )
                        
                        generate_preview_btn.click(
                            generate_character_preview,
                            inputs=[],
                            outputs=[preview_image, preview_status]
                        )
                        
                        regenerate_preview_btn.click(
                            regenerate_character_preview,
                            inputs=[],
                            outputs=[preview_image, preview_status]
                        )
            
            # Tab 2: Chat Interface
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            type="messages"
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

                def send_message(message, history):
                    # Initialize history as a list if it's None (Gradio default)
                    if history is None:
                        history = []

                    # Handle empty messages
                    if not message:
                        return history, "", None, "No message sent", []

                    try:
                        # Send request to the chat server
                        chat_response = requests.post(
                            "http://127.0.0.1:8000/chat",
                            json={"message": message}
                        )

                        # Check if the server responded successfully
                        if not chat_response.ok:
                            response_text = "Error: Failed to get response from server."
                            history = history + [
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": response_text}
                            ]
                            return history, "", None, "Error processing request", []

                        # Parse the server response
                        data = chat_response.json()
                        response_text = data.get("response", "No response received")
                        image_data = data.get("image_data")  # Base64-encoded image data
                        image_message = data.get("image_message")
                        image_history = data.get("image_history", [])

                        # Append messages in the correct format
                        history = history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": response_text}
                        ]

                        # Process the image if available
                        image_display = None
                        if image_data:
                            image_bytes = base64.b64decode(image_data)
                            image_display = Image.open(BytesIO(image_bytes))

                        # Prepare image gallery items
                        gallery_items = []
                        for img_entry in image_history:
                            if os.path.exists(os.path.join(OUTPUT_DIR, img_entry["path"])):
                                gallery_items.append(os.path.join(OUTPUT_DIR, img_entry["path"]))

                        return history, "", image_display, image_message or "No image generated", gallery_items

                    except Exception as e:
                        # Append error message in the correct format
                        history = history + [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": f"Error: {str(e)}"}
                        ]
                        return history, "", None, f"Error: {str(e)}", []

                def clear_chat_history():
                    try:
                        requests.post("http://127.0.0.1:8000/clear_chat")
                        return [], None, "Chat cleared", []
                    except Exception as e:
                        return [], None, f"Error clearing chat: {str(e)}", []
                        
                def regenerate_current_image():
                    try:
                        # Call the regenerate endpoint
                        response = requests.post("http://127.0.0.1:8000/regenerate_image")
                        if not response.ok:
                            return None, "Failed to regenerate image"
                            
                        data = response.json()
                        image_data = data.get("image_data")
                        if not image_data:
                            return None, data.get("message", "No image generated")
                            
                        # Convert base64 to image
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                        return image, data.get("message", "Image regenerated successfully")
                    except Exception as e:
                        return None, f"Error: {str(e)}"

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

    # Start the FastAPI server in a separate thread
    import uvicorn
    import threading

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": app, "host": "127.0.0.1", "port": 8000}
    )

    server_thread.daemon = True
    server_thread.start()

    # Launch the Gradio UI
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    create_ui()