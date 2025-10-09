# 🤖 multi-modal-labeler

A Streamlit app for automatic labelling and annotation of comments from multiple CSV files and images using deep learning, translation, and sentiment analysis.

# Features
1. ### Batch Annotation for Multiple CSVs

* Select a folder containing any number of CSV files (each with a "text" column).

* Dropdown menu to select individual CSV for annotation.

* Translate text to English (googletrans), analyze sentiment (TextBlob), and edit labels interactively.

* Save per-row labels or batch process all rows.

* Output annotation CSVs named after their source (e.g., file_labels.csv).

2. ### Image Annotation

* Auto-label images using pretrained ResNet-152 (ImageNet classes).

* Manually adjust the suggested label in the UI.

* Save labels either per-image or for all images in the folder.

3. ### Built-in Error Handling

* Graceful fallback if the "text" column is missing or a CSV is empty.

* Robust saving, even when annotation files don't exist yet.

# Installation
```
pip install streamlit torch textblob pillow googletrans
git clone https://github.com/nanda1634/multi-modal-labeler.git
cd multi-modal-labeler
```
# Main dependencies:

* streamlit

* pandas

* pillow

* torch

* torchvision

* textblob

* googletrans

# Folder Structure
```
multi-modal-labeler/
├── data/
│   ├── text/         # Place all your CSV files with a "text" column here
│   └── images/       # Place all JPG/JPEG/PNG image files here
├── annotations/      # Output folder for annotation CSVs ("text_labels.csv", "image_labels.csv")
└── main.py           # Your main Streamlit application
```

### How to Use

1. source data:

* save the source csv files in data/text/ folder and pictures in data/image/ folder

2. Run it
```
python -m streamlit run main.py
```

3. Text Annotation Mode:

* In the sidebar, select "Text".

* Pick any CSV from the dropdown.

* Step through comment entries, see auto-translated/sentiment labels.

* Edit the label as needed.

Save annotation for a single row ("Save Annotation") or process all comments in the current CSV ("Save Annotations for All").

Annotated results will be saved in annotations/text_labels.csv.

2. Image Annotation Mode:

* In the sidebar, select "Image".

* Application scans the images/ folder for photos.

* Each image is auto-labelled using a deep learning model.

* Review/change label and save for single images or batch process all.

### Typical Workflow
* Quickly scan and annotate hundreds/thousands of CSV comments from multiple files.

* Translate any language comment to English, score its sentiment, and export clean labels for ML training.

* Use SOTA neural networks to label image datasets with a single click.

### Recommended:
To Run Prefer using the virtual environment for reliability:
```
python -m venv {env-name}
{env-name}/scripts/activate
streamlit run main.py
```

### Credits:

* Streamlit

* Googletrans

* TextBlob
 
* PyTorch

## License

MIT License - Free for personal and commercial use

## Support

For issues or feature requests, [please open an issue.](https://github.com/nanda1634/multi-modal-labeler/issues)

Streamline your annotation workflow with automated translation, sentiment analysis, and deep learning.
