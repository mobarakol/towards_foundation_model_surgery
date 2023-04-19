import os
from PIL import Image
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from transformers import AutoProcessor, Blip2ForConditionalGeneration


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if "cuda" in device else torch.float32
model_type = "vit_h"
checkpoint = "./sam_vit_h_4b8939.pth"


model = sam_model_registry[model_type](checkpoint=checkpoint)
model.to(device)
predictor = SamPredictor(model)
mask_generator = SamAutomaticMaskGenerator(model)


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
captioning_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map="sequential", load_in_8bit=True
)


root_folder = "/media/mmlab/data/lalith/7_LLM/dataset/EndoVis-18-VQA/"
output_folder = "outputs_BLIP/"
# complete = ["seq_2", "seq_7", "seq_9", "seq_14", "seq_15"]
complete = []


for seq_folder in os.listdir(root_folder):
    if seq_folder in complete or not seq_folder.startswith("seq_"):
        continue
    for frame_file in os.listdir(os.path.join(root_folder, seq_folder, "left_frames")):
        if not frame_file.endswith(".png"):
            continue
        frame_number = frame_file.split(".")[0]
        image_path = os.path.join(root_folder, seq_folder, "left_frames", frame_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        qa_file = os.path.join(
            root_folder, seq_folder, "vqa2/Classification", f"{frame_number}_QA.txt"
        )

        if not os.path.exists(qa_file):
            continue

        with open(qa_file) as f:
            qa_pairs = f.readlines()

        output_dir_new = os.path.join(output_folder, seq_folder)
        if not os.path.exists(output_dir_new):
            os.makedirs(output_dir_new)

        with open(
            os.path.join(output_dir_new, f"{frame_number}_captions.txt"), "w"
        ) as f:
            for qa_pair in qa_pairs:
                questions, answer = qa_pair.strip().split("|")
                for question in questions.split("&"):
                    text_prompt = f"Question: {question} Answer:"
                    inputs = processor(
                        image, text=text_prompt, return_tensors="pt", verbose=False
                    ).to(device, torch_dtype)
                    out = captioning_model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(out[0], skip_special_tokens=True).strip()
                    print(f"{frame_number}: {question}|{caption}\n")
                    f.write(f"{question}|{caption}\n")
