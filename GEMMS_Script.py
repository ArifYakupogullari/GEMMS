import torch
import os
import cv2
import numpy as np
import json
import re
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# --- CONFIGURATION ---
MODEL_ID = "google/gemma-3-27b-it" 
INPUT_FOLDER = "/data/images/group_ab"
OUTPUT_FOLDER = "/data/results/group_ab"
LOG_FILE = os.path.join(OUTPUT_FOLDER, "audit_log.jsonl")
DEVICE = "cuda:0"

# =====================================================================
# --- USER ACTION REQUIRED: PROPRIETARY LOGIC PLACEHOLDERS ---
# The following three variables form the "Brain" and "Eyes" of the pipeline.
# They have been replaced with generic dummy text for this public repository.
# To achieve production-level, highly accurate safety audits, you MUST 
# rebuild these variables with your own domain-specific logic.
# =====================================================================

# --- 1. THE RETINA (SITE VOCABULARY) ---
# WHAT IT DOES: This list tells the SAM 3 model exactly what to look for. If an object isn't here, the system is blind to it.
# THE PROBLEM WITH THIS DUMMY: It uses generic terms like "worker" or "debris pile". SAM 3 struggles with ambiguity.
# HOW TO FIX IT: Build an exhaustive, hyper-specific list. Don't write "wood"; write "plywood sheet", "wooden pallet", "scrap wood offcut". The more specific the noun, the better the visual grounding.
SITE_VOCABULARY = [
    "guardrail", "scaffolding", "ladder", "floor opening",
    "wooden pallet", "debris pile", "electrical cable", "worker",
    "hard hat", "power tool"
]

# --- 2. THE BRAIN (SAFETY CODEX) ---
# WHAT IT DOES: The ultimate source of truth for the LLM. It defines a hazard vs. a safe state.
# THE PROBLEM WITH THIS DUMMY: It just says "Walkways must be clear." The LLM might see wood actively being built into a wall and hallucinate it as a "trip hazard."
# HOW TO FIX IT: Write strict conditional logic. Define "Visual Violations" and "Safe States." Teach the AI to apply contextual overrides (e.g., an active worker holding a tool next to a cable is SAFE; a cable snaking across an empty hallway is an ABANDONED HAZARD).
SITE_SAFETY_MANUAL = (
    "*** BASIC SITE SAFETY GUIDELINES ***\n\n"
    "1. Floor Integrity: Walkways and working areas must be clear of loose debris, tools, and trip hazards.\n"
    "2. Material Logistics: Materials like plywood or pipes must be secured from falling or blowing away.\n"
    "3. Edge Protection: All drop-offs, voids, and holes must have proper physical barriers or rigid covers.\n"
    "4. Electrical: Cables should be managed safely and not left snaking across active walking paths unattended."
)

# --- 3. THE LOGIC (AUDITOR PROTOCOL) ---
# WHAT IT DOES: Establishes the prompt engineering architecture.
# THE PROBLEM WITH THIS DUMMY: It just says "find hazards," leading to lazy, single-step thinking where the LLM just guesses based on the inventory list.
# HOW TO FIX IT: Implement a strict "Chain of Thought" sequence. Force the AI to write out an analysis BEFORE the JSON where it: 1) Identifies grouped structures, 2) Analyzes spatial relationships, 3) Looks for missing components (like missing kickboards), and 4) Formulates a verdict based on the Codex.
AUDITOR_PROMPT_CONTEXT = (
    "### 📘 REFERENCE MATERIAL\n"
    f"{SITE_SAFETY_MANUAL}\n\n"

    "### 🕵️‍♂️ AUDIT PROTOCOL\n"
    "Analyze the image and the provided visual inventory to identify potential safety hazards.\n"
    "Focus on missing safety equipment, unsecured materials, and contextual hazards (e.g., abandoned trip hazards).\n\n"

    "### OUTPUT FORMAT\n\n"
    "**### ANALYSIS:**\n"
    "[Provide a brief paragraph analyzing the scene. Identify the objects present and assess if any safety guidelines are being violated based on their context.]\n\n"
    "**### JSON OUTPUT:**\n"
    "```json\n"
    "[\n"
    "  {\n"
    "    \"violation\": \"[Short description of the hazard]\",\n"
    "    \"target_object\": \"[The specific object triggering the issue]\", \n"
    "    \"severity\": \"[CRITICAL / MEDIUM / LOW]\",\n"
    "    \"reasoning\": \"[Brief explanation of why this object and context creates a hazard]\"\n"
    "  }\n"
    "]\n"
    "```"
)
# =====================================================================


# --- UTILS & EXECUTION ---
def load_models():
    """Load Gemma and SAM3 models safely."""
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        print("CRITICAL ERROR: SAM3 library not found. Please install 'sam3' package.")
        exit(1)

    print("--- Loading Gemma (The Thinking Engine)... ---")
    brain_proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    brain_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

    print("--- Loading SAM3 (The Eyes)... ---")
    sam_model = build_sam3_image_model().to(DEVICE)
    sam_proc = Sam3Processor(sam_model)

    return brain_model, brain_proc, sam_model, sam_proc

def gemma_think(model, processor, image, prompt, max_tokens=15000):
    """Generate reasoning and JSON from the VLM."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2)
        return processor.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def draw_laser(cv_img, box, label, color=(0,0,255), is_mask=False, mask_data=None):
    """Draws the laser pointer annotation."""
    try:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Draw Mask if available
        if is_mask and mask_data is not None:
            overlay = cv_img.copy()
            overlay[mask_data] = color
            cv2.addWeighted(overlay, 0.25, cv_img, 0.75, 0, cv_img)

        # Draw Laser Dot
        cv2.circle(cv_img, (cx, cy), 8, color, -1)
        cv2.circle(cv_img, (cx, cy), 3, (255, 255, 255), -1)

        # Draw Label Background & Text
        display_label = label.replace("PEAB:", "").strip()
        font_scale = 0.6
        (w, h), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

        # Calculate box coords with bounds checking
        tx_min = max(0, cx - w // 2 - 4)
        ty_min = max(0, cy - 25 - h - 4)
        tx_max = min(cv_img.shape[1], cx + w // 2 + 4)
        ty_max = min(cv_img.shape[0], cy - 25 + 4)

        cv2.rectangle(cv_img, (tx_min, ty_min), (tx_max, ty_max), (0,0,0), -1)
        cv2.putText(cv_img, display_label, (cx - w // 2, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    except Exception as e:
        print(f"Warning: Failed to draw laser for {label}: {e}")

def log_json(data):
    try:
        with open(LOG_FILE, "a") as f: f.write(json.dumps(data) + "\n")
    except: pass

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    # Load Models
    brain, brain_proc, sam, sam_proc = load_models()
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))]

    for filename in tqdm(files):
        img_path = os.path.join(INPUT_FOLDER, filename)
        print(f"Processing: {filename}")

        try:
            pil_image = Image.open(img_path).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            inference_state = sam_proc.set_image(pil_image)

            # --- STEP 1: SENSING (Inventory) ---
            inventory = []
            for item in SITE_VOCABULARY:
                out = sam_proc.set_text_prompt(state=inference_state, prompt=item)
                if out["scores"].numel() > 0:
                    valid_indices = [i for i, s in enumerate(out["scores"].flatten()) if s > 0.18]
                    for idx in valid_indices:
                        inventory.append({
                            "object": item,
                            "confidence": out["scores"][idx].item(),
                            "box": out["boxes"][idx].cpu().tolist(),
                            "mask": out["masks"][idx][0].cpu().numpy().astype(bool)
                        })

            inv_text = "\n".join([f"- {x['object']} (Conf: {x['confidence']:.2f}) at coordinates {x['box']}" for x in inventory]) if inventory else "No recognized objects detected."

            # --- STEP 2: REASONING (Chain of Thought) ---
            FINAL_GEMMA_PROMPT = f"""
{AUDITOR_PROMPT_CONTEXT}

### 👁️ VISUAL INVENTORY (SAM 3 DETECTION LOG)
The following objects and their spatial coordinates have been detected in the image:
{inv_text}

### INSTRUCTIONS FOR THIS IMAGE
Execute the analysis on the provided image and visual inventory based on the safety material.
Begin your response with the ### ANALYSIS block.
"""
            response = gemma_think(brain, brain_proc, pil_image, FINAL_GEMMA_PROMPT)

            # Parse response for JSON block
            violations = []
            try:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[-2]

                violations = json.loads(json_str)
            except Exception:
                violations = []

            # --- STEP 3: VISUALIZATION ---
            has_detection = False
            final_violations = []

            for v in violations:
                target = v.get("target_object", "")
                violation_name = v.get("violation", "HAZARD")
                color = (0,0,255) if v.get("severity") == "CRITICAL" else (0,165,255)

                # A: Missing Item (Point at Anchor)
                if "missing" in target.lower() or "missing" in violation_name.lower():
                    anchor_found = False
                    for item in inventory:
                        valid_anchors = [
                            "guardrail", "Combisafe barrier", "scaffolding tower",
                            "floor opening", "core drill hole", "grid colander",
                            "lowered floor joist", "plywood sheet", "formwork panel",
                            "electrical panel"
                        ]
                        if item["object"] in valid_anchors:
                            draw_laser(cv_img, item["box"], f"MISSING: {violation_name}", (255, 0, 0), True, item["mask"])
                            anchor_found = True
                            break
                    if not anchor_found:
                         cv2.putText(cv_img, f"MISSING: {violation_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    has_detection = True
                    final_violations.append(v)
                    continue

                # B: Active Hazard (Point at Object)
                matched = next((x for x in inventory if x["object"] in target or target in x["object"]), None)
                if matched:
                    draw_laser(cv_img, matched["box"], violation_name, color, True, matched["mask"])
                    has_detection = True
                    final_violations.append(v)
                else:
                    # C: Re-sense Fallback
                    out_resense = sam_proc.set_text_prompt(state=inference_state, prompt=target)
                    if out_resense["scores"].numel() > 0:
                        best_idx = torch.argmax(out_resense["scores"]).item()
                        if out_resense["scores"][best_idx].item() > 0.1:
                            box = out_resense["boxes"][best_idx].cpu().tolist()
                            mask = out_resense["masks"][best_idx][0].cpu().numpy().astype(bool)
                            draw_laser(cv_img, box, violation_name, color, True, mask)
                            has_detection = True
                            final_violations.append(v)

            # Log
            log_entry = {"file": filename, "inventory": [x["object"] for x in inventory], "analysis": response.split("### JSON OUTPUT:")[0].strip(), "violations": final_violations}
            log_json(log_entry)

            if has_detection:
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"VERA_Audit_{filename}"), cv_img)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
