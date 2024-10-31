#Important code to execute
from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

app = FastAPI()

# Load the TFLite model
tflite_model_path = "pcm.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Potato___Early_blight","Potato___Late_blight","Potato___healthy"]

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Convert image to RGB if it's not already
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')
       
    
    # Resize image to the size expected by the model (e.g., 256x256)
    image = image.resize((256, 256))
    
    # Convert image to numpy array
    image_array = np.array(image, dtype=np.float32)  # Ensure float32 type
    # print(image_array)
    # plt.show()
    # Normalize image array
    # image_array = image_array / 256.0  # Adjust normalization if required by your model
    
    # Expand dimensions to match model input shape
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch

def read_file_as_images(data) -> dict:
    image = Image.open(BytesIO(data))
    image_batch = preprocess_image(image)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image_batch)
    interpreter.invoke()
    
    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process the results
    prediction_class = class_names[np.argmax(output_data[0])]
    confidence = np.max(output_data[0])
    
    return {
        'class': prediction_class,
        'confidence': float(confidence)
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        bytes = await file.read()
        result = read_file_as_images(bytes)
        return result
    except Exception as e:
        # Debugging: Print the error message
        print(f"Error encountered: {e}")
        # Log the error if needed
        with open('error_log.txt', 'a', encoding='utf-8') as f:
            f.write(f"Error encountered: {e}\n")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
