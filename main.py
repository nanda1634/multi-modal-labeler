import streamlit as st
import pandas as pd
import os
from textblob import TextBlob
from googletrans import Translator
from PIL import Image
import torch
from torchvision.models import resnet152
from torchvision import transforms
import asyncio
import pandas.errors
import glob

# ========== PATHS ==========
TEXT_PATH = "{path}" # complete Path of text folder
IMAGE_PATH = "{path}" # path of images folder
ANNOTATIONS_PATH = "{path}" # path of annotations folder
os.makedirs(ANNOTATIONS_PATH, exist_ok=True)

# ========== Load Models (One time) ==========
@st.cache_resource
def get_image_model():
    model = resnet152(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def get_labels():
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    import urllib.request
    response = urllib.request.urlopen(LABELS_URL)
    labels = [line.strip() for line in response.read().decode("utf-8").splitlines()]
    return labels

image_model = get_image_model()
labels = get_labels()
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== Translator ==========
translator = Translator()

# ========== Auto-label Functions ==========

def translate_text_sync(text):
    async def translate_async(text):
        try:
            translated = await Translator().translate(text, dest="en")
            return translated.text
        except Exception:
            return text
    return asyncio.run(translate_async(text))

def auto_label_text(text):
    translated = translate_text_sync(text)
    polarity = TextBlob(translated).sentiment.polarity
    if polarity > 0.5:
        label = "Positive"
    elif polarity < -0.5:
        label = "Negative"
    else:
        label = "Neutral"
    return label, translated

def auto_label_image(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = image_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = image_model(input_tensor)
        _, idx = torch.max(output, 1)
        return labels[idx.item()]


# ========== Streamlit UI ==========
st.title("🤖 Multi‑Modal Auto-Labeling Tool")

mode = st.sidebar.radio("Choose Data Type", ["Text", "Image"])

# ========== ASK USER FOR A CSV FOLDER ==========
CSV_FOLDER = st.text_input(
    "Folder containing CSV files (each with a comments column):",
    value="E:/New_folder/ML_A/data/text/"
)

# Scan for CSV files in the given folder
csv_files = []
if os.path.isdir(TEXT_PATH):
    csv_files = glob.glob(os.path.join(TEXT_PATH, "*.csv"))

if mode == "Text":
    st.header("📝 Text Auto‑Annotation")
    if not csv_files:
        st.warning("No CSV files found in the folder.")
    else:
        # Dropdown for user to pick which file to annotate
        selected_csv = st.selectbox("Choose a CSV file to annotate:", csv_files)
        df = pd.read_csv(selected_csv)
        if "text" not in df.columns:
            st.warning(f"The selected CSV ({os.path.basename(selected_csv)}) does not have a 'text' column.")
        else:
            idx = st.number_input("Text index", 0, len(df)-1, 0)
            text_item = df.iloc[idx]['text']
            st.write("Original Text:", text_item)
            label, translated = auto_label_text(text_item)
            st.write("Translated Text:", translated)
            label_input = st.text_input("Label", value=label)

            ann_file = os.path.join(
                ANNOTATIONS_PATH,
                os.path.basename(selected_csv).replace(".csv", "_labels.csv")
            )
        if st.button("Save Annotation"):
            ann_file = os.path.join(ANNOTATIONS_PATH, "text_labels.csv")
            columns = ["index", "text", "translated_text", "label"]
            try:
                ann_df = pd.read_csv(ann_file)
                ann_df = ann_df[ann_df["index"] != idx]
            except (FileNotFoundError, pandas.errors.EmptyDataError):
                ann_df = pd.DataFrame(columns=columns)
            new_row = {
                "index": idx,
                "text": text_item,
                "translated_text": translated,
                "label": label_input,
            }
            ann_df = pd.concat([ann_df, pd.DataFrame([new_row])], ignore_index=True)
            ann_df.to_csv(ann_file, index=False, encoding="utf-8")
            st.success(f"Annotation saved for index {idx}.")

        # --- Batch save ---
        if st.button("Save Annotations for All"):
            ann_file = os.path.join(ANNOTATIONS_PATH, "text_labels.csv")
            columns = ["index", "text", "translated_text", "label"]
            all_rows = []
            for i, row in df.iterrows():
                lbl, trans = auto_label_text(row['text'])
                lbl = lbl if lbl else 'Neutral'
                all_rows.append({
                    "index": i,
                    "text": row['text'],
                    "translated_text": trans,
                    "label": lbl
                })
            df_labels = pd.DataFrame(all_rows, columns=columns)
            df_labels.to_csv(ann_file, index=False, encoding="utf-8")
            st.success(f"Saved annotations for all {len(df)} text items.")

elif mode == "Image":
    st.header("🖼 Image Auto‑Annotation")
    if os.path.exists(IMAGE_PATH) and os.listdir(IMAGE_PATH):
        images = [f for f in os.listdir(IMAGE_PATH) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        idx = st.number_input("Image index", 0, len(images)-1, 0)
        img_file = os.path.join(IMAGE_PATH, images[idx])
        st.image(Image.open(img_file), caption=images[idx], width=300)
        auto_label = auto_label_image(img_file)
        label = st.text_input("Label", value=auto_label)

        if st.button("Save Annotation"):
            ann_file = os.path.join(ANNOTATIONS_PATH, "image_labels.csv")
            columns = ["filename", "label"]
            try:
                ann_df = pd.read_csv(ann_file)
                ann_df = ann_df[ann_df["filename"] != images[idx]]
            except (FileNotFoundError, pandas.errors.EmptyDataError):
                ann_df = pd.DataFrame(columns=columns)
            new_row = {
                "filename": images[idx],
                "label": label,
            }
            ann_df = pd.concat([ann_df, pd.DataFrame([new_row])], ignore_index=True)
            ann_df.to_csv(ann_file, index=False, encoding="utf-8")
            st.success(f"Annotation saved for {images[idx]}.")

        if st.button("Save Annotations for All"):
            ann_file = os.path.join(ANNOTATIONS_PATH, "image_labels.csv")
            all_rows = []
            for img_name in images:
                img_path = os.path.join(IMAGE_PATH, img_name)
                lbl = auto_label_image(img_path)
                all_rows.append({"filename": img_name, "label": lbl})
            df_labels = pd.DataFrame(all_rows, columns=["filename", "label"])
            df_labels.to_csv(ann_file, index=False)
            st.success(f"Saved annotations for all {len(images)} images.")
    else:

        st.warning("No images found in folder.")
