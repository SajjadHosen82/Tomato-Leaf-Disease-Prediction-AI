import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model
model = tf.keras.models.load_model('tomato_model_final.keras')

# 2. Class names
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]

# 3. Prediction function


def predict_disease(image):
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    img_array = img_array / 255.0

    # Make prediction (এখানে সব ক্লাসের পার্সেন্টেজ বের করা হচ্ছে)
    predictions = model.predict(img_array)[0]

    # Gradio Label এর জন্য একটি ডিকশনারি তৈরি করা (কোন রোগের কত পার্সেন্টেজ)
    confidences = {class_names[i]: float(
        predictions[i]) for i in range(len(class_names))}

    return confidences


# 4. Custom Theme Setup
custom_theme = gr.themes.Soft(
    primary_hue="green",
    neutral_hue="slate"
)

# 5. Custom Gradio Interface
demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="📸 Upload Tomato Leaf Image Here"),
    # আউটপুটটিকে Text এর বদলে Label করা হয়েছে, যা টপ ৩টি প্রেডিকশন দেখাবে
    outputs=gr.Label(num_top_classes=3,
                     label="🔍 AI Prediction Result (Confidence %)"),
    title="🍅 Tomato Leaf Disease Predictor",
    description="### 🌿 Live Demo Project \nEasily check if your tomato plant leaves have any diseases. Upload an image below to see the AI's prediction.",
    theme=custom_theme
)

# 6. Run the app
if __name__ == "__main__":
    demo.launch()
