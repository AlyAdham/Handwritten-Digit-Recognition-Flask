import seaborn as sns
import torch
import base64
# import config # Assuming config.py exists and has HOST, PORT, DEBUG_MODE
import matplotlib
import numpy as np
from PIL import Image
from io import BytesIO
# Import MnistModel from your train.py file
from train import MnistModel
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify # Ensure render_template is imported
import os # Import the os module
import torch.nn as nn # Import nn module for isinstance checks
import torch.nn.functional as F # Import F for functional operations
import re # Import regex for parsing


# --- Assuming config.py exists with HOST, PORT, DEBUG_MODE ---
# If you don't have a config.py, you'll need to define these variables here
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using default values for Flask.")
    class config:
        HOST = '127.0.0.1'
        PORT = 5000
        DEBUG_MODE = False
# -----------------------------------------------------------


matplotlib.use('Agg') # Use Agg backend for Matplotlib in a non-GUI environment

MODEL = None # Global variable to hold the loaded model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Determine device

app = Flask(__name__) # Initialize the Flask application


# --- Helper class to save intermediate layer outputs ---
class SaveOutput:
    def __init__(self):
        # Changed to a dictionary to store outputs by layer name or a key
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        # This __call__ method is not directly used with the current hook registration approach
        pass

    def clear(self):
        self.outputs = {} # Clear the dictionary
# ----------------------------------------------------


# --- Function to register hooks for the main page interpretability visualization ---
# This only hooks the first Conv2d layer to match the original interpretability_img function
def register_hook_for_main_viz():
    save_output = SaveOutput()
    hook_handles = []
    # Assuming MODEL is initialized and is an nn.Module
    if MODEL is not None:
        for name, module in MODEL.named_modules(): # Iterate through named modules
            # Only capture the output of the first Conv2d layer by name 'conv1A'
            if name == 'conv1A':
                # Store the output in the dictionary with a key
                handle = module.register_forward_hook(lambda m, i, o: save_output.outputs.update({'conv1A_output': o.detach().to('cpu')}))
                hook_handles.append(handle)
                break # Stop after finding conv1A
    # Note: We are not returning hook_handles here, which means we don't explicitly remove this hook.
    # For a single hook in a web app context, this might be acceptable, but for multiple hooks
    # or more complex scenarios, returning and removing handles is crucial for memory management.
    return save_output
# ---------------------------------------------------------------------------------


# --- Helper function to convert tensor output to numpy array ---
def module_output_to_numpy(tensor):
    # Ensure tensor is on CPU and detached before converting to numpy
    return tensor.detach().to('cpu').numpy()
# ----------------------------------------------------------


# --- Helper function for adding labels to bar charts ---
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rects in rects:
        height = rects.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rects.get_x() + rects.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# ------------------------------------------------------


# --- Function to generate the probability bar chart image ---
def prob_img(probs):
    fig, ax = plt.subplots()
    rects = ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    ax.set_ylim(0, 110) # Set y-limit slightly above 100 for labels
    ax.set_title('Probability % of Digit by Model')
    autolabel(rects, ax) # Add percentage labels on bars
    probimg = BytesIO()
    fig.savefig(probimg, format='png') # Save plot to a BytesIO object
    probencoded = base64.b64encode(probimg.getvalue()).decode('utf-8') # Encode image to base64
    plt.close(fig) # Close the figure to free up memory
    return probencoded
# --------------------------------------------------------


# --- Function to generate the interpretability image for the main page ---
# This visualizes the output of the first convolutional layer
def interpretability_img(save_output):
    # Access the output using the key 'conv1A_output'
    captured_output = save_output.outputs.get('conv1A_output')

    # Ensure captured_output is not None and is a tensor
    if captured_output is None or not isinstance(captured_output, torch.Tensor):
        print("Warning: No convolutional layer output captured for interpretability.")
        # Return a placeholder or empty string if no output is captured
        return ""

    # Assuming the captured_output is the tensor from conv1A
    images = module_output_to_numpy(captured_output)

    # Ensure there are at least 16 images to display
    if images.shape[1] < 16:
         print(f"Warning: Only {images.shape[1]} convolutional layer outputs available, expected at least 16 for display.")
         # Adjust the number of images to display if less than 16
         num_images_to_display = images.shape[1]
         L = int(np.ceil(np.sqrt(num_images_to_display)))
         W = L # Make the grid roughly square
    else:
         num_images_to_display = 16
         L = 4
         W = 4


    with plt.style.context("seaborn-v0_8-white"): # Use a consistent style
        fig, axes = plt.subplots(L, W, figsize=(W * 3, L * 3)) # Adjust figure size based on grid
        plt.suptitle("Interpretability by Model (First Conv Block Feature Maps)", fontsize=16) # Added label
        axes = axes.ravel() # Flatten the axes array

        for idx in range(num_images_to_display):
            # Check if the index is within the number of axes
            if idx < len(axes): # Ensure index is within bounds of axes array
                 # Display the feature map (assuming grayscale or single channel per subplot)
                 axes[idx].imshow(images[0, idx], cmap='viridis') # Using a colormap for better visualization
                 # Use idx in the title
                 axes[idx].set_title(f"Channel {idx+1}", fontsize=10) # Add title to each subplot
                 axes[idx].axis('off') # Turn off axes ticks and labels
            else:
                 # Hide extra subplots if fewer images are displayed than grid size
                 axes[idx].axis('off')


        # Hide any remaining unused subplots if the grid is larger than num_images_to_display
        for j in range(num_images_to_display, len(axes)):
              axes[j].axis('off')


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    interpretimg = BytesIO()
    fig.savefig(interpretimg, format='png') # Save plot to BytesIO
    interpretencoded = base64.b64encode(interpretimg.getvalue()).decode('utf-8') # Encode to base64
    plt.close(fig) # Close the figure to free up memory
    return interpretencoded
# ------------------------------------------------------------------------


# --- Function to generate Saliency Map ---
# Keep this function as it might still be useful later or for other explanations
def generate_saliency_map(img_tensor, predicted_class):
    # Ensure model is available and on the correct device
    if MODEL is None:
        print("Error: Model not loaded for saliency map generation.")
        return ""

    # We need gradients with respect to the input image
    # Clone the tensor and set requires_grad to True
    img_tensor = img_tensor.clone().requires_grad_(True).to(DEVICE)

    MODEL.eval() # Set model to evaluation mode
    # No need for torch.no_grad() here because we need gradients

    # Perform forward pass
    outputs = MODEL(x=img_tensor)

    # Get the score for the predicted class
    # outputs is log_softmax output, so we need the value at the predicted class index
    score = outputs[0, predicted_class]

    # Backward pass to get gradients of the score with respect to the input image
    # This will populate img_tensor.grad
    score.backward()

    # Get the gradients
    gradients = img_tensor.grad

    # Calculate saliency map (absolute values of gradients)
    # The gradients tensor shape is (1, 1, H, W) for grayscale
    saliency = gradients.abs().squeeze().cpu().numpy() # Get absolute values, remove batch/channel dims, move to cpu, convert to numpy

    # Normalize the saliency map for visualization (optional but often improves visual clarity)
    # Avoid division by zero if saliency is all zeros
    if np.max(saliency) > 0:
        saliency = saliency / np.max(saliency)
    else:
        saliency = np.zeros_like(saliency) # If all zeros, keep it as zeros

    # Visualize the saliency map
    fig, ax = plt.subplots(figsize=(5, 5))
    # Display the saliency map as a heatmap
    # Using 'hot' colormap often works well for saliency
    im = ax.imshow(saliency, cmap='hot', interpolation='nearest')
    ax.set_title(f'Saliency Map for Predicted Class {predicted_class}')
    ax.axis('off') # Turn off axes

    # Optional: Add a colorbar to show the scale
    # fig.colorbar(im)

    plt.tight_layout()
    saliency_img_bytes = BytesIO()
    fig.savefig(saliency_img_bytes, format='png') # Save plot to BytesIO
    saliency_encoded = base64.b64encode(saliency_img_bytes.getvalue()).decode('utf-8') # Encode to base64
    plt.close(fig) # Close the figure

    return saliency_encoded
# ----------------------------------------


# --- Function to perform MNIST prediction ---
# Modified to also return the predicted class for the visualization page
def mnist_prediction(img):
    save_output = register_hook_for_main_viz() # Register hook for the main page viz
    img = img.to(DEVICE, dtype=torch.float) # Move image tensor to the correct device

    # Need to ensure MODEL is not None before calling it
    if MODEL is None:
        print("Error: Model is not loaded.")
        return "Error: Model not loaded", "", "", "", None # Added extra returns for saliency and predicted_class


    MODEL.eval() # Set model to evaluation mode before inference
    # Note: We need gradients for saliency map, so we cannot use torch.no_grad() here
    # for the forward pass that precedes saliency calculation.
    # However, the forward pass in this function is primarily for getting the prediction
    # and probability. The saliency map function will perform its own forward pass
    # with gradient tracking enabled on a cloned tensor.
    with torch.no_grad(): # Use no_grad for the prediction and probability calculation part
        outputs = MODEL(x=img) # Perform forward pass

    # Ensure outputs is a tensor before processing
    if not isinstance(outputs, torch.Tensor):
         print("Error: Model did not return a tensor.")
         return "Error: Prediction failed", "", "", "", None # Added extra returns

    # Calculate probabilities and generate probability image
    probs = torch.exp(outputs.data)[0] * 100
    probencoded = prob_img(probs)

    # Generate interpretability image for the main page
    interpretencoded = interpretability_img(save_output)

    # Get the predicted class (digit)
    _, predicted_class_tensor = torch.max(outputs.data, 1)
    predicted_class = predicted_class_tensor.item() # Get the scalar predicted class

    # Generate Saliency Map for the predicted class (optional, keep for now)
    saliency_encoded = generate_saliency_map(img, predicted_class) # Pass the original image tensor and predicted class

    return predicted_class, probencoded, interpretencoded, saliency_encoded, img # Return all results including original image tensor
# -----------------------------------------


# --- Route for the main drawing and prediction page ---
# Global variables to store the last processed image and predicted class for visualization on other pages
last_processed_img_tensor = None
last_predicted_class = None
last_prob_chart_encoded = None # Store the probability chart as well

@app.route("/process", methods=["GET", "POST"])
def process():
    global last_processed_img_tensor, last_predicted_class, last_prob_chart_encoded # Declare intent to modify global variables
    data_url = str(request.get_data()) # Get image data from the request
    offset = data_url.index(',')+1 # Find the start of the base64 data
    img_bytes = base64.b64decode(data_url[offset:]) # Decode base64 data
    img = Image.open(BytesIO(img_bytes)) # Open image using PIL
    img = img.convert('L') # Convert to grayscale
    img = img.resize((28, 28)) # Resize to 28x28 pixels
    # img.save(r'templates\image.png') # Commented out to avoid saving files unnecessarily
    img_np = np.array(img) # Convert to NumPy array
    img_np = img_np.reshape((1, 28, 28)) # Reshape to (channels, height, width) - 1 channel for grayscale
    img_tensor = torch.tensor(img_np, dtype=torch.float).unsqueeze(0) # Convert to PyTorch tensor and add batch dimension

    # Store the processed image tensor globally for visualization page
    last_processed_img_tensor = img_tensor.clone() # Store a clone of the tensor


    # Perform the prediction and get all visualization data
    predicted_class, probencoded, interpretencoded, saliency_encoded, _ = mnist_prediction(img_tensor)

    # Store the predicted class and probability chart globally
    last_predicted_class = predicted_class
    last_prob_chart_encoded = probencoded


    # Prepare the JSON response
    response = {
        'data': str(predicted_class), # Return the predicted class
        'probencoded': probencoded, # probencoded is already string
        'interpretencoded': interpretencoded, # interpretencoded is already string
        'saliencyencoded': saliency_encoded # Include the saliency map image
    }
    return jsonify(response) # Return JSON response
# ----------------------------------------------------


# --- Route for the main drawing and prediction page ---
@app.route("/", methods=["GET", "POST"])
def start():
    # Render the main drawing page template
    return render_template("default.html")
# -----------------------------------


# --- Function to parse the confusion matrix from the text file ---
def parse_confusion_matrix(metrics_content):
    confusion_matrix_data = None
    # Look for the "Confusion Matrix:" section
    cm_match = re.search(r"Confusion Matrix:\s*\n([\s\S]*?)(?=Classification Report:|\Z)", metrics_content)
    if cm_match:
        cm_text = cm_match.group(1).strip()
        # Split the text into lines and then split each line into numbers
        # Handle potential header/index lines if they exist in the file
        lines = cm_text.split('\n')
        matrix_rows = []
        for line in lines:
            # Use regex to find numbers (integers or floats) in the line
            numbers = re.findall(r'\d+', line) # Assuming integers in confusion matrix
            if numbers:
                matrix_rows.append([int(n) for n in numbers])

        if matrix_rows:
            confusion_matrix_data = np.array(matrix_rows)

    return confusion_matrix_data
# --------------------------------------------------------------


# --- Function to parse the classification report from the text file ---
def parse_classification_report(metrics_content):
    report_data = {}
    # Look for the "Classification Report:" section
    cr_match = re.search(r"Classification Report:\s*\n([\s\S]*)", metrics_content)
    if cr_match:
        cr_text = cr_match.group(1).strip()
        lines = cr_text.split('\n')
        # Skip header line and potentially the last lines (accuracy, macro avg, weighted avg)
        # Assuming the format is consistent: class precision recall f1-score support
        # We need to be careful about parsing the floating point numbers

        # Regex to capture class, precision, recall, f1-score, support
        # Handles potential extra spaces and floating point numbers
        # Example line: ' 0       0.99      0.99      0.99      1000'
        # This regex looks for a digit (the class), then captures groups of non-whitespace
        # characters for precision, recall, f1-score, and support.
        # It also handles potential leading/trailing whitespace on lines.
        line_pattern = re.compile(r'^\s*(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)\s*$')


        for line in lines:
            match = line_pattern.match(line)
            if match:
                class_label = match.group(1)
                precision = float(match.group(2))
                recall = float(match.group(3))
                f1_score = float(match.group(4))
                support = int(match.group(5))

                report_data[class_label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1_score,
                    'support': support
                }

    return report_data
# -------------------------------------------------------------------


# --- Function to visualize the confusion matrix as a heatmap ---
def visualize_confusion_matrix(confusion_matrix_data):
    if confusion_matrix_data is None or confusion_matrix_data.size == 0:
        return "" # Return empty string if no data

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png')
    img_encoded = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_encoded
# -----------------------------------------------------------

# --- Function to visualize classification report metrics as bar charts ---
def visualize_classification_report(report_data):
    if not report_data:
        return "", "", "", "" # Return empty strings if no data

    classes = sorted(report_data.keys(), key=int) # Sort classes numerically
    precision_values = [report_data[c]['precision'] for c in classes]
    recall_values = [report_data[c]['recall'] for c in classes]
    f1_values = [report_data[c]['f1-score'] for c in classes]
    support_values = [report_data[c]['support'] for c in classes] # Keep support values

    # Precision Bar Chart
    fig_p, ax_p = plt.subplots(figsize=(10, 5))
    rects_p = ax_p.bar(classes, precision_values, color='skyblue')
    ax_p.set_ylim(0, 1.1) # Metrics are typically between 0 and 1
    ax_p.set_title('Precision per Class')
    ax_p.set_xlabel('Digit Class')
    ax_p.set_ylabel('Precision')
    autolabel(rects_p, ax_p) # Add labels on bars
    plt.tight_layout()
    img_bytes_p = BytesIO()
    fig_p.savefig(img_bytes_p, format='png')
    precision_image = base64.b64encode(img_bytes_p.getvalue()).decode('utf-8')
    plt.close(fig_p)

    # Recall Bar Chart
    fig_r, ax_r = plt.subplots(figsize=(10, 5))
    rects_r = ax_r.bar(classes, recall_values, color='lightgreen')
    ax_r.set_ylim(0, 1.1)
    ax_r.set_title('Recall per Class')
    ax_r.set_xlabel('Digit Class')
    ax_r.set_ylabel('Recall')
    autolabel(rects_r, ax_r) # Add labels on bars
    plt.tight_layout()
    img_bytes_r = BytesIO()
    fig_r.savefig(img_bytes_r, format='png')
    recall_image = base64.b64encode(img_bytes_r.getvalue()).decode('utf-8')
    plt.close(fig_r)

    # F1-Score Bar Chart
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 5))
    rects_f1 = ax_f1.bar(classes, f1_values, color='salmon')
    ax_f1.set_ylim(0, 1.1)
    ax_f1.set_title('F1-Score per Class')
    ax_f1.set_xlabel('Digit Class')
    ax_f1.set_ylabel('F1-Score')
    autolabel(rects_f1, ax_f1) # Add labels on bars
    plt.tight_layout()
    img_bytes_f1 = BytesIO()
    fig_f1.savefig(img_bytes_f1, format='png')
    f1_image = base64.b64encode(img_bytes_f1.getvalue()).decode('utf-8')
    plt.close(fig_f1)

    # Support values can be returned as text or a simple structure if needed
    # For now, let's just return the images
    return precision_image, recall_image, f1_image, support_values # Returning support values as list


# --- New route for the performance metrics page ---
@app.route('/metrics')
def view_metrics():
    # Define the path to the performance metrics file
    metrics_file_path = os.path.join('checkpoint', 'performance_metrics.txt')
    metrics_content = "Metrics file not found."
    confusion_matrix_image = ""
    precision_image = ""
    recall_image = ""
    f1_image = ""
    support_values = []
    error_message = None


    # Read the content of the metrics file
    try:
        with open(metrics_file_path, 'r') as f:
            metrics_content = f.read()

        # Parse and visualize the confusion matrix
        confusion_matrix_data = parse_confusion_matrix(metrics_content)
        if confusion_matrix_data is not None:
            confusion_matrix_image = visualize_confusion_matrix(confusion_matrix_data)
        else:
             error_message = "Could not parse Confusion Matrix data from the metrics file."


        # Parse and visualize the classification report
        report_data = parse_classification_report(metrics_content)
        if report_data:
            precision_image, recall_image, f1_image, support_values = visualize_classification_report(report_data)
        else:
             if error_message: # Append to existing error message
                 error_message += " Could not parse Classification Report data from the metrics file."
             else:
                 error_message = "Could not parse Classification Report data from the metrics file."


    except FileNotFoundError:
        error_message = f"Error: The metrics file was not found at {metrics_file_path}. Please run train.py first to generate it."
        confusion_matrix_image = "" # Clear image on error
        precision_image = ""
        recall_image = ""
        f1_image = ""
        support_values = []

    except Exception as e:
        error_message = f"An error occurred while reading or parsing the metrics file: {e}"
        confusion_matrix_image = "" # Clear image on error
        precision_image = ""
        recall_image = ""
        f1_image = ""
        support_values = []


    # Pass the visualizations and support values to the metrics.html template
    return render_template('metrics.html',
                           confusion_matrix_image=confusion_matrix_image,
                           precision_image=precision_image,
                           recall_image=recall_image,
                           f1_image=f1_image,
                           support_values=support_values, # Pass support values
                           error_message=error_message) # Pass any error message
# -----------------------------------------------


# --- Function to capture outputs for the CNN/MLP visualization page ---
# Modified to capture outputs after pool1 and pool2 specifically
def capture_three_key_layer_outputs(img):
     captured_data = {} # Dictionary to store {layer_name: output_tensor}
     hook_handles_viz = [] # List to store hook handles

     if MODEL is None:
          print("Error: Model is not loaded for visualization capture.")
          return {} # Return empty dict if model is not loaded

     # Define the specific layers we want to visualize outputs from
     layers_to_hook = ['pool1', 'pool2']

     # Create a hook function that stores the output with the layer name
     def create_hook(layer_name):
         def hook(module, module_in, module_out):
             # Detach and move to CPU immediately
             captured_data[layer_name] = module_out.detach().to('cpu')
         return hook

     # Register hooks on the specific named layers
     if MODEL is not None:
          for name, module in MODEL.named_modules():
               if name in layers_to_hook: # Hook layers defined in layers_to_hook
                   handle = module.register_forward_hook(create_hook(name))
                   hook_handles_viz.append(handle)
                   # print(f"Registered key viz hook on layer: {name}") # Optional print


     MODEL.eval() # Set model to evaluation mode
     with torch.no_grad(): # Disable gradient calculation
         # Need to perform a forward pass to trigger the hooks
         # The output of this forward pass is not used, we only care about the hook outputs
         _ = MODEL(x=img.to(DEVICE, dtype=torch.float))

     # Remove hooks after capturing outputs to prevent memory leaks and unwanted behavior
     for handle in hook_handles_viz:
          handle.remove()

     # Return the dictionary of captured data
     return captured_data
# --------------------------------------------------------------------


# --- Helper function to visualize a tensor and return a base64 image ---
# This function will now be used for the specific three visualizations
# It will visualize the average across channels for conv/pool outputs
def visualize_averaged_feature_map(tensor, title="Visualization"):
    # This function is specifically for visualizing the average of feature maps
    # from a conv/pool layer.

    if tensor.ndim == 3: # Assuming (C, H, W) after removing batch dim
         # Calculate the average across the channel dimension (dim 0)
         averaged_map = torch.mean(tensor, dim=0).cpu().numpy()

         fig, ax = plt.subplots(figsize=(5, 5))
         # Display the averaged map as a heatmap
         im = ax.imshow(averaged_map, cmap='viridis', interpolation='nearest') # Using viridis colormap
         ax.set_title(title, fontsize=14) # Set title
         ax.axis('off') # Turn off axes

         # Optional: Add a colorbar
         # fig.colorbar(im)

         plt.tight_layout()
         img_bytes = BytesIO()
         fig.savefig(img_bytes, format='png') # Save plot to BytesIO
         img_encoded = base64.b64encode(img_bytes.getvalue()).decode('utf-8') # Encode to base64
         plt.close(fig) # Close the figure to free up memory
         return img_encoded

    else:
         # This function is intended for feature maps (C, H, W), return error for other shapes
         return "Unsupported tensor shape for visualization (expected C, H, W)."
# -----------------------------------------------------------------------


# --- Helper function to visualize the input image with predicted label ---
def visualize_input_with_prediction(img_tensor, predicted_class):
    # Assuming img_tensor is (1, 1, H, W)
    if img_tensor.ndim != 4 or img_tensor.shape[0] != 1 or img_tensor.shape[1] != 1:
         return "Unsupported image tensor shape for visualization."

    # Get the single image array (H, W)
    img_np = img_tensor.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np, cmap='gray') # Display the grayscale image
    ax.set_title(f'Predicted Digit: {predicted_class}', fontsize=16) # Add the predicted label as title
    ax.axis('off') # Turn off axes

    plt.tight_layout()
    img_bytes = BytesIO()
    fig.savefig(img_bytes, format='png') # Save plot to BytesIO
    img_encoded = base64.b64encode(img_bytes.getvalue()).decode('utf-8') # Encode to base64
    plt.close(fig) # Close the figure

    return img_encoded
# -------------------------------------------------------------------


# --- New route for the CNN/MLP visualization page ---
# This page will display three key visualizations
@app.route('/cnn_mlp_viz')
def cnn_mlp_viz():
    # Retrieve the last processed image tensor and predicted class
    global last_processed_img_tensor, last_predicted_class, last_prob_chart_encoded
    if last_processed_img_tensor is None or last_predicted_class is None:
        # If no image has been processed yet, return an informative message
        return render_template('cnn_mlp_viz.html', viz_stage1='', viz_stage2='', viz_predicted_number='', viz_probability_chart='', message="Please draw a digit and click predict first to generate visualizations.")


    # Capture the intermediate layer outputs for the key layers (pool1 and pool2)
    captured_outputs = capture_three_key_layer_outputs(last_processed_img_tensor)

    # Prepare the visualization data dictionary
    viz_data = {}

    # 1. Visualization after the first convolutional block (pool1) - Fused
    if 'pool1' in captured_outputs:
         # Use the new visualize_averaged_feature_map function
         viz_data['viz_stage1'] = visualize_averaged_feature_map(captured_outputs['pool1'][0], title="Stage 1: Early Features (Fused)")
    else:
         viz_data['viz_stage1'] = "Visualization for pool1 not available."


    # 2. Visualization after the second convolutional block (pool2) - Fused
    if 'pool2' in captured_outputs:
         # Use the new visualize_averaged_feature_map function
         viz_data['viz_stage2'] = visualize_averaged_feature_map(captured_outputs['pool2'][0], title="Stage 2: Complex Features (Fused)")
    else:
         viz_data['viz_stage2'] = "Visualization for pool2 not available."

    # 3. Visualization of the predicted number (input image with label and probability chart)
    # We can combine the input image visualization and the probability chart
    input_viz_encoded = visualize_input_with_prediction(last_processed_img_tensor, last_predicted_class)
    viz_data['viz_predicted_number'] = input_viz_encoded
    viz_data['viz_probability_chart'] = last_prob_chart_encoded # Use the stored probability chart


    # Pass the visualization data (base64 images) to the HTML template
    # We will pass them individually for easier display in the template
    return render_template('cnn_mlp_viz.html',
                           viz_stage1=viz_data.get('viz_stage1', ''),
                           viz_stage2=viz_data.get('viz_stage2', ''),
                           viz_predicted_number=viz_data.get('viz_predicted_number', ''),
                           viz_probability_chart=viz_data.get('viz_probability_chart', ''))
# ---------------------------------------------------


# --- Main execution block ---
if __name__ == "__main__":
    # --- Model Loading ---
    # Ensure MnistModel class is accessible before loading the model
    try:
        # Assuming MnistModel is defined in train.py and train.py is in the same directory
        # from train import MnistModel # This import is already at the top

        MODEL = MnistModel(classes=10) # Instantiate the model
        # Check if the model checkpoint file exists
        model_checkpoint_path = 'checkpoint/mnist.pt'
        if os.path.exists(model_checkpoint_path):
            # Load the saved model state dictionary
            MODEL.load_state_dict(torch.load(
                model_checkpoint_path, map_location=DEVICE))
            print(f"Model loaded successfully from {model_checkpoint_path}")
        else:
            print(f"Warning: Model checkpoint not found at {model_checkpoint_path}. Please run train.py to train the model.")
            # If model file is not found, set MODEL to None and print a warning
            MODEL = None

        # If the model was successfully loaded, move it to the device and set to evaluation mode
        if MODEL is not None:
            MODEL.to(DEVICE)
            MODEL.eval() # Set model to evaluation mode

    except ImportError:
        print("Error: Could not import MnistModel from train. Please ensure train.py exists and is in the correct location.")
        MODEL = None # Set MODEL to None if import fails
    except Exception as e:
        print(f"Error loading the model: {e}")
        MODEL = None # Set MODEL to None if any other error occurs during loading

    # --- Flask App Run ---
    # Ensure config variables are available before running the app
    if 'config' not in locals() or not hasattr(config, 'HOST') or not hasattr(config, 'PORT') or not hasattr(config, 'DEBUG_MODE'):
        print("Warning: config.py or its required variables not fully available. Using default Flask run parameters.")
        app.run(debug=True) # Use Flask's default run with debug enabled
    else:
        # Run the Flask application with configurations from config.py
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)
# -----------------------
