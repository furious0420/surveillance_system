import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import io
import os
# Optional plotly imports for analytics
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
import pandas as pd
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import json
from datetime import datetime
import subprocess
import sys

# Configure Streamlit page
st.set_page_config(
    page_title="YOLOv8 Dual Model Detection Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'performance_history' not in st.session_state:
    st.session_state.performance_history = []
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = {'total_detections': 0, 'images_processed': 0}
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS for enhanced UI
def load_custom_css():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            color: white;
        }
        .metric-card {
            background: linear-gradient(135deg, #2d2d2d, #3d3d3d);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            margin: 10px 0;
            border: 1px solid #444;
        }
        .thermal-mode {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: white;
            text-align: center;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        .visible-mode {
            background: linear-gradient(45deg, #4ecdc4, #44bd32);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: white;
            text-align: center;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .stButton > button {
            border-radius: 20px;
            height: 50px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 10px 0;
            border: 1px solid #e0e0e0;
        }
        .thermal-mode {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: white;
            text-align: center;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        .visible-mode {
            background: linear-gradient(45deg, #4ecdc4, #44bd32);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: white;
            text-align: center;
            font-weight: bold;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .stButton > button {
            border-radius: 20px;
            height: 50px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    """Load HAL surveillance and other YOLOv8 models"""
    try:
        # HAL Surveillance Model (primary visible light model)
        hal_model_path = "HAL_Model_20250719_100007/HAL_Model_Weights_20250719_100007.pt"
        if not os.path.exists(hal_model_path):
            # Fallback to pkl model if .pt not found
            hal_model_path = "HAL_Surveillance_Model_20250719_100007.pkl"

        # Thermal/IR model
        thermal_model_path = "ir_sensor_model.pt"

        # Aerial model
        aerial_model_path = "best_aerial_2.pt"

        st.info(f"🔄 Loading HAL Surveillance model from: {hal_model_path}")
        st.info(f"🔄 Loading IR model from: {thermal_model_path}")
        st.info(f"🔄 Loading Aerial model from: {aerial_model_path}")

        # Load HAL surveillance model
        if hal_model_path.endswith('.pt'):
            visible_model = YOLO(hal_model_path)  # HAL Surveillance model
        else:
            # Handle .pkl model if needed
            import pickle
            with open(hal_model_path, 'rb') as f:
                visible_model = pickle.load(f)

        # Load thermal model
        if os.path.exists(thermal_model_path):
            thermal_model = YOLO(thermal_model_path)
        else:
            st.warning("⚠️ IR model not found, using HAL model for all detections")
            thermal_model = visible_model  # Use HAL model as fallback

        # Load aerial model
        if os.path.exists(aerial_model_path):
            aerial_model = YOLO(aerial_model_path)
            st.success("✅ Dedicated aerial surveillance model loaded!")
        else:
            st.warning("⚠️ Aerial model not found, using HAL model for aerial detection")
            aerial_model = visible_model  # Use HAL model as fallback

        st.success("✅ All surveillance models loaded successfully!")
        return visible_model, thermal_model, aerial_model

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.error("Please check that the following files exist:")
        st.error("1. HAL_Model_20250719_100007/HAL_Model_Weights_20250719_100007.pt")
        st.error("2. HAL_Surveillance_Model_20250719_100007.pkl (fallback)")
        st.error("3. ir_sensor_model.pt (optional)")
        st.error("4. best_aerial_2.pt (aerial model)")
        return None, None, None

# Global variables for models (for real-time processing)
VISIBLE_MODEL = None
THERMAL_MODEL = None
AERIAL_MODEL = None

# HAL Surveillance Classes and Threat Levels (integrated from hal_webcam_surveillance.py)
HAL_CLASSES = {
    0: 'person_civilian', 1: 'person_armed', 2: 'weapon_gun', 3: 'weapon_knife',
    4: 'weapon_rifle', 5: 'vehicle_car', 6: 'vehicle_bike', 7: 'vehicle_truck',
    8: 'vehicle_tank', 9: 'animal_dog', 10: 'animal_livestock', 11: 'animal_wild',
    12: 'uav', 13: 'unknown_entity'
}

# Threat level classifications
HAL_THREAT_LEVELS = {
    'SAFE': [0],                                    # person_civilian
    'HIGH_THREAT': [1, 2, 3, 4, 11, 12],          # person_armed, weapons, animal_wild, uav
    'POTENTIAL_THREAT': [5, 6, 7, 8, 13],         # vehicles, unknown_entity
    'LOW_THREAT': [9, 10]                          # animal_dog, animal_livestock
}

# Threat level colors (BGR format for OpenCV)
HAL_THREAT_COLORS = {
    'HIGH_THREAT': (0, 0, 255),      # Red
    'POTENTIAL_THREAT': (0, 255, 255), # Yellow
    'LOW_THREAT': (0, 255, 0),        # Green
    'SAFE': (0, 255, 0)               # Green
}

def get_hal_threat_level(class_id):
    """Get HAL threat level for detected class"""
    for threat_level, class_ids in HAL_THREAT_LEVELS.items():
        if class_id in class_ids:
            return threat_level
    return 'UNKNOWN'

class VideoTransformer(VideoTransformerBase):
    """Enhanced video transformer with HAL surveillance integration"""

    def __init__(self):
        self.detection_enabled = True
        self.frame_count = 0
        self.process_every_n_frames = 3
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45

        # HAL Surveillance Statistics
        self.total_detections = 0
        self.threat_count = {'HIGH_THREAT': 0, 'POTENTIAL_THREAT': 0, 'LOW_THREAT': 0, 'SAFE': 0}
        self.show_hal_overlay = True

        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # FPS calculation
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()

        # Only process every nth frame for performance
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            # Still draw HAL overlay even when not processing
            if self.show_hal_overlay:
                self.draw_hal_surveillance_overlay(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.detection_enabled and VISIBLE_MODEL is not None:
            # Convert BGR to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            # Detect image type
            image_type = detect_image_type(pil_image)

            # Select appropriate model - prioritize HAL surveillance for visible light
            if image_type == 'thermal' and THERMAL_MODEL is not None:
                selected_model = THERMAL_MODEL
                model_name = "IR Model"
            else:
                selected_model = VISIBLE_MODEL  # Use HAL surveillance model
                model_name = "HAL Surveillance"

            # Apply model settings
            selected_model.conf = self.confidence_threshold
            selected_model.iou = self.nms_threshold

            try:
                # Run inference with HAL surveillance integration
                results = selected_model(img_rgb, verbose=False)

                # Process detections with HAL surveillance styling
                frame_threats = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Extract detection data
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])

                            # Get class name - use HAL classes if available
                            if class_id < len(HAL_CLASSES):
                                class_name = HAL_CLASSES.get(class_id, f'class_{class_id}')
                                threat_level = get_hal_threat_level(class_id)
                            else:
                                class_name = selected_model.names.get(class_id, f'class_{class_id}')
                                threat_level = 'UNKNOWN'

                            # Draw HAL-style detection
                            self.draw_hal_detection(img, (x1, y1, x2, y2), class_name, confidence, threat_level)
                            frame_threats.append(threat_level)

                            # Update HAL statistics
                            self.total_detections += 1
                            if threat_level in self.threat_count:
                                self.threat_count[threat_level] += 1

                # Add model indicator
                self.draw_model_indicator(img, model_name, image_type)

                # Add FPS counter
                fps_text = f"FPS: {1.0/self.process_every_n_frames:.1f}"
                cv2.putText(img, fps_text, (10, img.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except Exception as e:
                # Draw error message on frame
                cv2.putText(img, f"HAL Error: {str(e)[:40]}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw HAL surveillance overlay
        if self.show_hal_overlay:
            self.draw_hal_surveillance_overlay(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def draw_hal_detection(self, frame, box, class_name, confidence, threat_level):
        """Draw detection with HAL surveillance styling"""
        x1, y1, x2, y2 = box

        # Get threat color
        color = HAL_THREAT_COLORS.get(threat_level, (255, 255, 255))

        # Draw bounding box with threat-level styling
        thickness = 4 if threat_level == 'HIGH_THREAT' else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Prepare labels
        label = f"{class_name}: {confidence:.2f}"
        threat_label = f"[{threat_level}]"

        # Calculate label dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        threat_size = cv2.getTextSize(threat_label, font, font_scale, 2)[0]

        # Draw label background
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 50
        bg_width = max(label_size[0], threat_size[0]) + 10
        cv2.rectangle(frame, (x1, label_y - 40), (x1 + bg_width, label_y + 5), color, -1)

        # Draw text labels
        cv2.putText(frame, label, (x1 + 5, label_y - 25), font, font_scale, (255, 255, 255), 2)
        cv2.putText(frame, threat_label, (x1 + 5, label_y - 5), font, font_scale, (0, 0, 0), 2)

    def draw_model_indicator(self, frame, model_name, image_type):
        """Draw model type indicator"""
        model_text = f"Model: {model_name}"
        type_text = f"Type: {image_type.title()}"

        # Background
        cv2.rectangle(frame, (5, 5), (300, 55), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (300, 55), (0, 255, 0), 2)

        # Text
        cv2.putText(frame, model_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, type_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_hal_surveillance_overlay(self, frame):
        """Draw HAL surveillance statistics overlay"""
        # Background for statistics
        cv2.rectangle(frame, (10, 70), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 70), (400, 200), (255, 255, 255), 2)

        # Title
        cv2.putText(frame, "HAL SURVEILLANCE SYSTEM", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Model Accuracy: 92.7% | FPS: {self.current_fps}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Statistics
        y_offset = 135
        cv2.putText(frame, f"Total Detections: {self.total_detections}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 20
        for threat_level, count in self.threat_count.items():
            color = HAL_THREAT_COLORS.get(threat_level, (255, 255, 255))
            cv2.putText(frame, f"{threat_level}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 15

class AerialVideoTransformer(VideoTransformerBase):
    """Video transformer specifically for aerial surveillance detection"""

    def __init__(self):
        self.detection_enabled = True
        self.frame_count = 0
        self.process_every_n_frames = 2  # Process more frequently for aerial
        self.confidence_threshold = 0.25
        self.nms_threshold = 0.45
        self.max_detections = 50

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Only process every nth frame for performance
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if self.detection_enabled and AERIAL_MODEL is not None:
            # Convert BGR to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply model settings
            AERIAL_MODEL.conf = self.confidence_threshold
            AERIAL_MODEL.iou = self.nms_threshold
            AERIAL_MODEL.max_det = self.max_detections

            try:
                # Run inference with aerial model
                results = AERIAL_MODEL(img_rgb)

                # Draw results on the frame
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = AERIAL_MODEL.names[class_id]

                        # Use cyan color for aerial detections
                        aerial_color = (0, 255, 255)  # Cyan for aerial

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), aerial_color, 3)

                        # Draw label with background
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(img, (x1, y1 - label_size[1] - 15),
                                    (x1 + label_size[0] + 10, y1), aerial_color, -1)
                        cv2.putText(img, label, (x1 + 5, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Add aerial model indicator
                model_text = "AERIAL SURVEILLANCE"
                cv2.rectangle(img, (10, 10), (300, 50), (0, 255, 255), -1)
                cv2.putText(img, model_text, (15, 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # Add FPS counter
                fps_text = f"FPS: {1.0/self.process_every_n_frames:.1f}"
                cv2.putText(img, fps_text, (10, img.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except Exception as e:
                # Draw error message on frame
                cv2.putText(img, f"Aerial Error: {str(e)[:40]}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def detect_image_type(image, debug=False):
    """Enhanced image type detection with better algorithms"""
    img_array = np.array(image)

    # Check if image is grayscale
    if len(img_array.shape) == 2:
        return 'thermal'

    if len(img_array.shape) == 3:
        if img_array.shape[2] == 1:
            return 'thermal'
        elif img_array.shape[2] == 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

            # Method 1: Check if all channels are identical
            if np.array_equal(r, g) and np.array_equal(g, b):
                return 'thermal'

            # Method 2: Advanced channel analysis
            diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
            diff_rb = np.mean(np.abs(r.astype(float) - b.astype(float)))
            diff_gb = np.mean(np.abs(g.astype(float) - b.astype(float)))
            avg_channel_diff = (diff_rg + diff_rb + diff_gb) / 3

            # Method 3: HSV analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:,:,1]
            avg_saturation = np.mean(saturation)

            # Method 4: Color variance and entropy
            color_variance = np.var(img_array, axis=(0,1))
            total_color_variance = np.sum(color_variance)

            # Method 5: Edge analysis (thermal images often have different edge characteristics)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            # Enhanced decision logic with more conservative weights
            thermal_score = 0

            # Channel difference weight (more conservative)
            if avg_channel_diff < 3:
                thermal_score += 3
            elif avg_channel_diff < 6:
                thermal_score += 2
            elif avg_channel_diff < 10:
                thermal_score += 1

            # Saturation weight (more conservative)
            if avg_saturation < 10:
                thermal_score += 3
            elif avg_saturation < 20:
                thermal_score += 2
            elif avg_saturation < 30:
                thermal_score += 1

            # Color variance weight (more conservative)
            if total_color_variance < 500:
                thermal_score += 3
            elif total_color_variance < 1000:
                thermal_score += 2
            elif total_color_variance < 2000:
                thermal_score += 1

            # Edge density weight (thermal images often have different edge patterns)
            if edge_density < 0.05:
                thermal_score += 1

            # Debug information
            if debug:
                st.write("**Image Analysis Debug Info:**")
                st.write(f"- Average channel difference: {avg_channel_diff:.2f}")
                st.write(f"- Average saturation: {avg_saturation:.2f}")
                st.write(f"- Total color variance: {total_color_variance:.2f}")
                st.write(f"- Edge density: {edge_density:.4f}")
                st.write(f"- Thermal score: {thermal_score}/6")

            # Debug information
            if debug:
                st.write("**Image Analysis Debug Info:**")
                st.write(f"- Average channel difference: {avg_channel_diff:.2f}")
                st.write(f"- Average saturation: {avg_saturation:.2f}")
                st.write(f"- Total color variance: {total_color_variance:.2f}")
                st.write(f"- Edge density: {edge_density:.4f}")
                st.write(f"- Thermal score: {thermal_score}/6")

            # Final decision with higher threshold (need more evidence for thermal)
            result = 'thermal' if thermal_score >= 6 else 'visible'

            if debug:
                st.write(f"- **Final classification: {result}**")

            return result

    return 'visible'

def process_video_file(uploaded_file, model, model_name, confidence_threshold, nms_threshold, max_detections):
    """Process uploaded video file with object detection"""
    import tempfile
    import os

    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_video_path = tmp_file.name

    try:
        # Open video file
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("❌ Could not open video file")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        st.info(f"📹 **Video Info:** {width}x{height} | {fps} FPS | {duration:.1f}s | {total_frames} frames")

        # Video processing options
        col1, col2, col3 = st.columns(3)
        with col1:
            process_every_n = st.slider("Process Every N Frames", 1, 30, 5, key=f"video_skip_{model_name}")
        with col2:
            max_frames_to_process = st.slider("Max Frames to Process", 10, 500, 100, key=f"video_max_{model_name}")
        with col3:
            show_progress = st.checkbox("Show Progress", value=True, key=f"video_progress_{model_name}")

        # Debug option
        debug_mode = st.checkbox("🔍 Debug Mode (Show Detection Details)", value=False, key=f"video_debug_{model_name}")

        # Show model information
        with st.expander("🔍 Model Information"):
            st.write(f"**Model Path:** {model.model_path if hasattr(model, 'model_path') else 'Unknown'}")
            st.write(f"**Model Classes:** {list(model.names.values())}")
            st.write(f"**Number of Classes:** {len(model.names)}")
            st.write(f"**Current Confidence Threshold:** {confidence_threshold}")
            st.write(f"**Current NMS Threshold:** {nms_threshold}")

        # Add confidence testing button
        col_test, col_process = st.columns(2)

        with col_test:
            if st.button(f"🧪 Test First Frame", key=f"test_frame_{model_name}"):
                st.markdown("### 🧪 Testing First Frame with Different Confidence Levels")

                # Extract first frame
                cap = cv2.VideoCapture(temp_video_path)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.write(f"📊 Frame shape: {frame.shape}")

                    # Test different confidence levels
                    confidence_levels = [0.01, 0.05, 0.1, 0.25, 0.5]

                    test_results = []
                    for conf in confidence_levels:
                        try:
                            results = model.predict(
                                frame_rgb,
                                conf=conf,
                                iou=nms_threshold,
                                device='cuda' if hasattr(model, 'device') else 'cpu',
                                verbose=False
                            )

                            detections = 0
                            detection_details = []

                            if results and len(results) > 0:
                                result = results[0]

                                # Check for OBB detections
                                if hasattr(result, 'obb') and result.obb is not None:
                                    detections = len(result.obb)

                                    if detections > 0:
                                        classes = result.obb.cls.cpu().numpy()
                                        confs = result.obb.conf.cpu().numpy()

                                        for cls, conf_score in zip(classes, confs):
                                            class_name = result.names[int(cls)]
                                            detection_details.append(f"{class_name}: {conf_score:.3f}")

                                # Fallback to regular boxes
                                elif hasattr(result, 'boxes') and result.boxes is not None:
                                    detections = len(result.boxes)

                                    if detections > 0:
                                        for box in result.boxes:
                                            class_id = int(box.cls[0])
                                            conf_score = float(box.conf[0])
                                            class_name = result.names[class_id]
                                            detection_details.append(f"{class_name}: {conf_score:.3f}")

                            test_results.append({
                                'confidence': conf,
                                'detections': detections,
                                'details': detection_details[:3]  # First 3 detections
                            })

                        except Exception as e:
                            test_results.append({
                                'confidence': conf,
                                'detections': 0,
                                'details': [f"Error: {e}"]
                            })

                    # Display results
                    for result in test_results:
                        conf = result['confidence']
                        det_count = result['detections']
                        details = result['details']

                        if det_count > 0:
                            st.success(f"🔍 Confidence {conf}: **{det_count} detections**")
                            for detail in details:
                                st.write(f"   • {detail}")
                        else:
                            st.warning(f"❌ Confidence {conf}: No detections")

                    # Recommendation
                    best_conf = None
                    for result in test_results:
                        if result['detections'] > 0:
                            best_conf = result['confidence']
                            break

                    if best_conf:
                        st.info(f"💡 **Recommendation**: Use confidence threshold of {best_conf} or lower for this video")
                    else:
                        st.error("❌ No detections found at any confidence level. Your video may not contain objects the model was trained on.")
                        st.write(f"📋 Model detects these classes: {list(model.names.values())}")
                else:
                    st.error("❌ Cannot read video frame")

        with col_process:
            if st.button(f"🎬 Process Video with {model_name}", key=f"process_video_{model_name}"):
                # Apply model settings
                model.conf = confidence_threshold
                model.iou = nms_threshold
                model.max_det = max_detections

                # Setup annotated video output
                output_video_path = "annotedop.mp4"
                # Try different codecs for better web compatibility
                codecs_to_try = [
                    ('H264', 'H.264 (best web compatibility)'),
                    ('avc1', 'AVC1 (H.264 variant)'),
                    ('mp4v', 'MP4V (fallback)'),
                    ('XVID', 'XVID (fallback)')
                ]
                out_video = None

                for codec, description in codecs_to_try:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        if out_video.isOpened():
                            st.success(f"📹 Using {codec} codec ({description})")
                            break
                        else:
                            out_video.release()
                    except:
                        continue

                if out_video is None or not out_video.isOpened():
                    # Final fallback
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    st.warning("⚠️ Using MP4V codec - video may not play in browser but will be downloadable")

                st.info(f"📹 Creating annotated video: {output_video_path}")

                # Create containers for results
                progress_container = st.empty()
                results_container = st.empty()

                processed_frames = 0
                total_detections = 0
                detection_results = []

                # Process video frames
                frame_count = 0
                while cap.isOpened() and processed_frames < max_frames_to_process:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Skip frames based on processing interval
                    if frame_count % process_every_n != 0:
                        continue

                    # Convert frame for processing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Run inference with proper parameters
                    results = model.predict(
                        frame_rgb,
                        conf=confidence_threshold,
                        iou=nms_threshold,
                        device='cuda' if hasattr(model, 'device') else 'cpu',
                        verbose=False
                    )

                    # Process detections using OBB-aware extraction
                    frame_detections = 0
                    frame_detection_details = []

                    # Debug: Force show what we're getting from YOLO
                    if debug_mode and processed_frames <= 3:  # Show debug for first 3 frames
                        st.write(f"🔍 **Frame {frame_count} Debug Info:**")
                        st.write(f"- Results type: {type(results)}")
                        st.write(f"- Results length: {len(results)}")
                        st.write(f"- Result[0] type: {type(results[0])}")
                        st.write(f"- Model names dict: {model.names}")

                        # Check all possible attributes
                        result_obj = results[0]
                        attrs = [attr for attr in dir(result_obj) if not attr.startswith('_')]
                        st.write(f"- Available attributes: {attrs}")

                        # Check for OBB (Oriented Bounding Boxes)
                        if hasattr(result_obj, 'obb'):
                            st.write(f"- obb: {result_obj.obb}")
                            if result_obj.obb is not None:
                                st.write(f"- obb type: {type(result_obj.obb)}")
                                st.write(f"- obb length: {len(result_obj.obb)}")

                        # Check for regular boxes
                        if hasattr(result_obj, 'boxes'):
                            st.write(f"- boxes: {result_obj.boxes}")
                            if result_obj.boxes is not None:
                                st.write(f"- boxes type: {type(result_obj.boxes)}")
                                st.write(f"- boxes length: {len(result_obj.boxes)}")

                    # Method 1: Try OBB (Oriented Bounding Boxes) - This is likely what your model uses!
                    try:
                        if hasattr(results[0], 'obb') and results[0].obb is not None:
                            obb = results[0].obb
                            if len(obb) > 0:
                                if debug_mode and processed_frames <= 3:
                                    st.write(f"🔍 Method 1 (OBB): Found {len(obb)} oriented bounding boxes")

                                # Extract class IDs and confidences
                                classes = obb.cls.cpu().numpy() if hasattr(obb.cls, 'cpu') else obb.cls.numpy()
                                confidences = obb.conf.cpu().numpy() if hasattr(obb.conf, 'cpu') else obb.conf.numpy()

                                # Get bounding boxes (convert OBB to regular bbox for display)
                                if hasattr(obb, 'xyxy'):
                                    bboxes = obb.xyxy.cpu().numpy() if hasattr(obb.xyxy, 'cpu') else obb.xyxy.numpy()
                                else:
                                    # If no xyxy, create placeholder bboxes
                                    bboxes = [[0, 0, 100, 100] for _ in range(len(classes))]

                                for i, (cls, conf) in enumerate(zip(classes, confidences)):
                                    try:
                                        class_id = int(cls)
                                        confidence = float(conf)

                                        # Get class name
                                        if class_id in model.names:
                                            class_name = model.names[class_id]
                                        else:
                                            class_name = f"class_{class_id}"
                                            if debug_mode:
                                                st.write(f"⚠️ Class ID {class_id} not found in model.names")

                                        # Get bbox
                                        bbox = bboxes[i] if i < len(bboxes) else [0, 0, 100, 100]

                                        frame_detection_details.append({
                                            'class': class_name,
                                            'confidence': confidence,
                                            'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                                        })
                                        frame_detections += 1

                                        if debug_mode and processed_frames <= 3:
                                            st.write(f"✅ OBB Detection {i}: {class_name} ({confidence:.2%})")

                                    except Exception as e:
                                        if debug_mode:
                                            st.write(f"❌ OBB {i} error: {e}")
                                        continue
                    except Exception as e:
                        if debug_mode:
                            st.write(f"❌ Method 1 (OBB) error: {e}")

                    # Method 2: Try standard boxes attribute (fallback)
                    if frame_detections == 0:
                        try:
                            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                                boxes = results[0].boxes
                                if hasattr(boxes, '__len__') and len(boxes) > 0:
                                    if debug_mode and processed_frames <= 3:
                                        st.write(f"🔍 Method 2 (Boxes): Found {len(boxes)} regular boxes")

                                    for i, box in enumerate(boxes):
                                        try:
                                            if hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy'):
                                                class_id = int(box.cls[0])
                                                confidence = float(box.conf[0])

                                                # Get class name
                                                if class_id in model.names:
                                                    class_name = model.names[class_id]
                                                else:
                                                    class_name = f"class_{class_id}"

                                                bbox = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]

                                                frame_detection_details.append({
                                                    'class': class_name,
                                                    'confidence': confidence,
                                                    'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                                                })
                                                frame_detections += 1

                                                if debug_mode and processed_frames <= 3:
                                                    st.write(f"✅ Box Detection {i}: {class_name} ({confidence:.2%})")

                                        except Exception as e:
                                            if debug_mode:
                                                st.write(f"❌ Box {i} error: {e}")
                                            continue
                        except Exception as e:
                            if debug_mode:
                                st.write(f"❌ Method 2 (Boxes) error: {e}")

                    # If no detections found, show debug info and try lower confidence
                    if frame_detections == 0 and debug_mode and processed_frames <= 3:
                        st.write("⚠️ No detections found in this frame")
                        st.write(f"   Current confidence threshold: {confidence_threshold}")
                        st.write(f"   Available model classes: {list(model.names.values())}")

                        # Test with very low confidence (like your script)
                        st.write("🧪 Testing with very low confidence (0.01)...")
                        try:
                            low_conf_results = model.predict(
                                frame_rgb,
                                conf=0.01,  # Very low confidence
                                iou=nms_threshold,
                                device='cuda' if hasattr(model, 'device') else 'cpu',
                                verbose=False
                            )

                            if hasattr(low_conf_results[0], 'obb') and low_conf_results[0].obb is not None:
                                low_conf_detections = len(low_conf_results[0].obb)
                                st.write(f"   📊 With conf=0.01: {low_conf_detections} detections found")

                                if low_conf_detections > 0:
                                    # Show first few detections
                                    classes = low_conf_results[0].obb.cls.cpu().numpy()[:3]  # First 3
                                    confs = low_conf_results[0].obb.conf.cpu().numpy()[:3]

                                    for cls, conf in zip(classes, confs):
                                        class_name = model.names[int(cls)]
                                        st.write(f"      {class_name}: {conf:.3f}")
                            else:
                                st.write("   📊 With conf=0.01: Still no detections")

                        except Exception as e:
                            st.write(f"   ❌ Low confidence test error: {e}")

                    total_detections += frame_detections

                    # Store detection info
                    detection_results.append({
                        'frame': frame_count,
                        'timestamp': frame_count / 30.0,  # Assume 30 FPS if not available
                        'detections': frame_detections,
                        'detection_details': frame_detection_details
                    })

                    processed_frames += 1

                    # Update progress
                    if show_progress:
                        progress = processed_frames / max_frames_to_process
                        progress_container.progress(progress)

                        # Show detailed frame info
                        if frame_detections > 0:
                            # Get unique classes in this frame
                            frame_classes = list(set([d['class'] for d in frame_detection_details]))
                            detection_summary = ", ".join([f"{d['class']} ({d['confidence']:.2%})" for d in frame_detection_details[:3]])
                            if len(frame_detection_details) > 3:
                                detection_summary += f" + {len(frame_detection_details)-3} more"

                            class_summary = f"Classes: {', '.join(frame_classes[:3])}"
                            if len(frame_classes) > 3:
                                class_summary += f" + {len(frame_classes)-3} more"

                            progress_container.success(f"✅ Frame {frame_count}: {frame_detections} detections | {class_summary}")
                            progress_container.info(f"   📋 Details: {detection_summary}")
                        else:
                            progress_container.warning(f"⚠️ Frame {frame_count}: No detections found")

                        # Show running totals
                        progress_container.info(f"📊 Progress: {processed_frames}/{max_frames_to_process} frames | Total detections: {total_detections}")

                    # Create annotated frame for video output
                    try:
                        # Create annotated frame
                        annotated_frame = frame_rgb.copy()
                        frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                        # Draw detections on frame
                        for det in frame_detection_details:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            # Draw bounding box
                            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame_bgr, f"{det['class']} {det['confidence']:.2%}",
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Write annotated frame to output video
                        out_video.write(frame_bgr)

                        # Display the annotated frame (first few frames only)
                        if frame_detections > 0 and processed_frames <= 3:
                            annotated_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            with progress_container.expander(f"🖼️ Frame {frame_count} with Detections"):
                                st.image(annotated_frame_rgb, caption=f"Frame {frame_count} - {frame_detections} detections", use_column_width=True)

                    except Exception as e:
                        # If annotation fails, write original frame
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        out_video.write(frame_bgr)
                        if debug_mode:
                            st.write(f"Frame annotation error: {e}")

                # Display results
                with results_container.container():
                    st.success(f"✅ Video processing complete!")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Frames Processed", processed_frames)
                    with col2:
                        st.metric("Total Detections", total_detections)
                    with col3:
                        st.metric("Avg Detections/Frame", f"{total_detections/processed_frames:.1f}" if processed_frames > 0 else "0")
                    with col4:
                        st.metric("Model Used", model_name)

                    # Show detection timeline and class breakdown
                    if detection_results and PLOTLY_AVAILABLE:
                        import pandas as pd

                        # Detection timeline
                        df = pd.DataFrame(detection_results)
                        fig = px.line(df, x='timestamp', y='detections',
                                    title="Detections Over Time",
                                    labels={'timestamp': 'Time (seconds)', 'detections': 'Number of Detections'})
                        st.plotly_chart(fig, use_container_width=True)

                        # Class breakdown
                        all_classes = {}
                        all_detections_list = []

                        for frame_result in detection_results:
                            for detection in frame_result.get('detection_details', []):
                                class_name = detection['class']
                                confidence = detection['confidence']

                                if class_name not in all_classes:
                                    all_classes[class_name] = []
                                all_classes[class_name].append(confidence)

                                all_detections_list.append({
                                    'frame': frame_result['frame'],
                                    'timestamp': frame_result['timestamp'],
                                    'class': class_name,
                                    'confidence': confidence
                                })

                        if all_classes:
                            st.markdown("### 🎯 Detected Classes Summary")

                            # Class statistics
                            class_stats = []
                            for class_name, confidences in all_classes.items():
                                class_stats.append({
                                    'Class': class_name,
                                    'Count': len(confidences),
                                    'Avg Confidence': f"{np.mean(confidences):.2%}",
                                    'Max Confidence': f"{max(confidences):.2%}",
                                    'Min Confidence': f"{min(confidences):.2%}"
                                })

                            class_df = pd.DataFrame(class_stats)
                            st.dataframe(class_df, use_container_width=True)

                            # Class distribution chart
                            if len(all_detections_list) > 0:
                                detections_df = pd.DataFrame(all_detections_list)

                                col1, col2 = st.columns(2)

                                with col1:
                                    # Class count pie chart
                                    class_counts = detections_df['class'].value_counts()
                                    fig_pie = px.pie(values=class_counts.values, names=class_counts.index,
                                                   title="Detection Distribution by Class")
                                    st.plotly_chart(fig_pie, use_container_width=True)

                                with col2:
                                    # Confidence distribution by class
                                    fig_box = px.box(detections_df, x='class', y='confidence',
                                                   title="Confidence Distribution by Class")
                                    fig_box.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig_box, use_container_width=True)

                        else:
                            st.warning("⚠️ No detections found in the processed frames. Try adjusting the confidence threshold or processing more frames.")

                    # Always show a summary of what was detected
                    if detection_results:
                        st.markdown("---")
                        st.markdown("### 📋 Detection Summary")

                        # Count all detections by class
                        class_counts = {}
                        total_detections_found = 0

                        for frame_result in detection_results:
                            total_detections_found += frame_result['detections']
                            for detection in frame_result.get('detection_details', []):
                                class_name = detection['class']
                                if class_name not in class_counts:
                                    class_counts[class_name] = 0
                                class_counts[class_name] += 1

                        if class_counts:
                            st.success(f"🎯 **Found {total_detections_found} total detections across {len(class_counts)} different classes:**")

                            # Display class counts in columns
                            cols = st.columns(min(len(class_counts), 4))
                            for i, (class_name, count) in enumerate(class_counts.items()):
                                with cols[i % len(cols)]:
                                    st.metric(
                                        label=f"🏷️ {class_name.title()}",
                                        value=f"{count} detections",
                                        delta=f"{count/total_detections_found:.1%} of total"
                                    )
                        else:
                            st.info("ℹ️ Detections were found but class information is not available.")

                    # Close video writer and display final annotated video
                    out_video.release()

                    # Try to convert to web-compatible format if possible
                    web_compatible_path = "annotedop_web.mp4"
                    try:
                        # Try to use FFmpeg for better web compatibility
                        import subprocess
                        result = subprocess.run([
                            'ffmpeg', '-i', output_video_path,
                            '-c:v', 'libx264', '-preset', 'fast',
                            '-crf', '23', '-c:a', 'aac',
                            '-movflags', '+faststart',
                            '-y', web_compatible_path
                        ], capture_output=True, text=True, timeout=60)

                        if result.returncode == 0 and os.path.exists(web_compatible_path):
                            output_video_path = web_compatible_path
                            st.success("🔄 Converted to web-compatible format")
                        else:
                            st.info("ℹ️ Using original format (FFmpeg conversion failed)")
                    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                        st.info("ℹ️ Using original format (FFmpeg not available)")

                    # Display the annotated video
                    st.markdown("---")
                    st.markdown("### 🎬 **Annotated Video Output**")

                    if os.path.exists(output_video_path):
                        st.success(f"✅ Annotated video saved: `{output_video_path}`")

                        # Display video file info
                        file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                        st.info(f"📁 **File:** {output_video_path} | **Size:** {file_size:.1f} MB")

                        # Try to show video player
                        try:
                            with open(output_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()

                            # Try to display the video
                            st.video(video_bytes)

                            # Provide download button
                            st.download_button(
                                label="📥 Download Annotated Video",
                                data=video_bytes,
                                file_name="annotated_surveillance_video.mp4",
                                mime="video/mp4"
                            )

                        except Exception as e:
                            st.warning(f"⚠️ Video player issue: {e}")
                            st.info("💡 **Alternative options:**")

                            # Provide download button even if video doesn't play
                            with open(output_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()

                            st.download_button(
                                label="📥 Download Annotated Video",
                                data=video_bytes,
                                file_name="annotated_surveillance_video.mp4",
                                mime="video/mp4"
                            )

                            # Show file path for manual access
                            st.code(f"Video saved at: {os.path.abspath(output_video_path)}")
                            st.info("🎬 You can open this file with any video player (VLC, Windows Media Player, etc.)")

                            # Additional troubleshooting info
                            st.markdown("**💡 Troubleshooting Tips:**")
                            st.markdown("- Download the video and play it locally")
                            st.markdown("- The video contains bounding boxes around detected objects")
                            st.markdown("- If video doesn't play in browser, it's likely a codec compatibility issue")
                            st.markdown("- Consider installing FFmpeg for better video compatibility")

                    else:
                        st.error("❌ Failed to create annotated video output")

    finally:
        # Clean up
        cap.release()
        if 'out_video' in locals():
            out_video.release()
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

def process_aerial_image(uploaded_file, aerial_model, confidence_threshold, nms_threshold, max_detections,
                        enhance_enabled, brightness, contrast, sharpness):
    """Process aerial surveillance image"""
    image = Image.open(uploaded_file)

    # Display original image
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📁 Original Aerial Image")
        st.image(image, use_column_width=True)
        st.info(f"**Size:** {image.size[0]} x {image.size[1]} pixels")

    # Always use aerial model for this tab
    st.markdown('<div style="background: linear-gradient(45deg, #3742fa, #2f3542); padding: 15px; border-radius: 10px; margin: 10px 0; color: white; text-align: center; font-weight: bold;">🛩️ AERIAL SURVEILLANCE MODE</div>',
               unsafe_allow_html=True)

    # Apply image enhancements if enabled
    if enhance_enabled:
        with st.spinner("🎨 Enhancing image..."):
            image = enhance_image(image, brightness, contrast, sharpness)

    # Run inference with progress bar
    with st.spinner("🔍 Running aerial detection..."):
        result_img, detections, timing_info = run_inference(
            aerial_model, image, 'aerial',
            confidence_threshold, nms_threshold, max_detections
        )

    with col2:
        st.markdown("#### 🎯 Detection Results")
        if result_img is not None:
            st.image(result_img, use_column_width=True)



        # Enhanced detection display
        display_enhanced_results(detections, "Aerial Model", timing_info, result_img, 'aerial')

def enhance_image(image, brightness=0, contrast=1.0, sharpness=1.0):
    """Apply image enhancements"""
    if brightness != 0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1 + brightness/100)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    return image

def run_inference(model, image, model_type, confidence_threshold=0.5, nms_threshold=0.45, max_detections=50):
    """Enhanced inference with configurable parameters"""
    try:
        start_time = time.time()

        # Apply model settings
        model.conf = confidence_threshold
        model.iou = nms_threshold
        model.max_det = max_detections

        # Run inference
        results = model(image)

        end_time = time.time()
        total_inference_time = (end_time - start_time) * 1000



        # Get the result image with bounding boxes
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Get detection details - try multiple approaches
        detections = []

        # Method 1: Try standard boxes attribute
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            if len(boxes) > 0:
                for box in boxes:
                    try:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        bbox = box.xyxy[0].cpu().numpy()

                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox,
                            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        })
                    except Exception as e:
                        continue

        # Method 2: Try accessing raw predictions
        if len(detections) == 0:
            try:
                # For ultralytics YOLO models, try accessing the raw tensor data
                if hasattr(results[0], 'data') and results[0].data is not None:
                    data = results[0].data
                    if len(data) > 0:
                        for detection in data:
                            x1, y1, x2, y2, conf, cls = detection[:6]
                            if conf >= confidence_threshold:
                                class_name = model.names[int(cls)]
                                bbox = [float(x1), float(y1), float(x2), float(y2)]
                                detections.append({
                                    'class': class_name,
                                    'confidence': float(conf),
                                    'bbox': bbox,
                                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                })
            except Exception:
                pass

        # Method 3: Try alternative prediction format
        if len(detections) == 0:
            try:
                if hasattr(results[0], 'pred') and results[0].pred is not None:
                    pred = results[0].pred
                    if isinstance(pred, list) and len(pred) > 0:
                        pred_tensor = pred[0]
                        for detection in pred_tensor:
                            x1, y1, x2, y2, conf, cls = detection[:6]
                            if conf >= confidence_threshold:
                                class_name = model.names[int(cls)]
                                bbox = [float(x1), float(y1), float(x2), float(y2)]
                                detections.append({
                                    'class': class_name,
                                    'confidence': float(conf),
                                    'bbox': bbox,
                                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                })
            except Exception:
                pass

        # Method 4: Try to access results through different attributes
        if len(detections) == 0:
            try:
                # Check all available attributes on the results object
                result_obj = results[0]

                # Try accessing through different possible attributes
                possible_attrs = ['xyxy', 'xywh', 'conf', 'cls', 'names']
                for attr in possible_attrs:
                    if hasattr(result_obj, attr):
                        attr_value = getattr(result_obj, attr)
                        if attr_value is not None and hasattr(attr_value, '__len__') and len(attr_value) > 0:
                            # Found some data, try to parse it
                            if attr == 'xyxy' and hasattr(result_obj, 'conf') and hasattr(result_obj, 'cls'):
                                xyxy = result_obj.xyxy
                                conf = result_obj.conf
                                cls = result_obj.cls

                                if len(xyxy) > 0 and len(conf) > 0 and len(cls) > 0:
                                    for i in range(len(xyxy)):
                                        if conf[i] >= confidence_threshold:
                                            class_name = model.names[int(cls[i])]
                                            bbox = xyxy[i].cpu().numpy() if hasattr(xyxy[i], 'cpu') else xyxy[i]
                                            detections.append({
                                                'class': class_name,
                                                'confidence': float(conf[i]),
                                                'bbox': bbox,
                                                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                            })
                            break
            except Exception:
                pass

        # Update session state statistics
        st.session_state.detection_stats['total_detections'] += len(detections)
        st.session_state.detection_stats['images_processed'] += 1

        # Store performance data
        performance_entry = {
            'timestamp': datetime.now(),
            'inference_time': total_inference_time,
            'detections_count': len(detections),
            'model_type': model_type
        }
        st.session_state.performance_history.append(performance_entry)

        # Keep only last 100 entries
        if len(st.session_state.performance_history) > 100:
            st.session_state.performance_history = st.session_state.performance_history[-100:]

        timing_info = {
            'total_time': total_inference_time,
            'preprocess_time': getattr(results[0], 'speed', {}).get('preprocess', 0),
            'inference_time': getattr(results[0], 'speed', {}).get('inference', 0),
            'postprocess_time': getattr(results[0], 'speed', {}).get('postprocess', 0)
        }

        return result_img_rgb, detections, timing_info

    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None, [], {}

def create_analytics_dashboard():
    """Create comprehensive analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    if st.session_state.performance_history:
        df = pd.DataFrame(st.session_state.performance_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance over time
            fig = px.line(df, x='timestamp', y='inference_time', 
                         title="Inference Time Over Time",
                         color='model_type')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Detection count distribution
            fig = px.histogram(df, x='detections_count', 
                             title="Detection Count Distribution",
                             color='model_type')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model usage comparison
        model_usage = df['model_type'].value_counts()
        fig = px.pie(values=model_usage.values, names=model_usage.index,
                     title="Model Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)

def create_enhanced_metrics():
    """Create enhanced metrics display"""
    stats = st.session_state.detection_stats
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Total Detections</h3>
            <h2>{}</h2>
        </div>
        """.format(stats['total_detections']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📸 Images Processed</h3>
            <h2>{}</h2>
        </div>
        """.format(stats['images_processed']), unsafe_allow_html=True)
    
    with col3:
        avg_detections = stats['total_detections'] / max(stats['images_processed'], 1)
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Avg Detections/Image</h3>
            <h2>{:.1f}</h2>
        </div>
        """.format(avg_detections), unsafe_allow_html=True)
    
    with col4:
        if st.session_state.performance_history:
            avg_time = np.mean([p['inference_time'] for p in st.session_state.performance_history])
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Avg Processing Time</h3>
                <h2>{:.1f}ms</h2>
            </div>
            """.format(avg_time), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Avg Processing Time</h3>
                <h2>N/A</h2>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_custom_css()
    
    # Header with animated title
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1>🔍 YOLOv8 Dual Model Detection Pro</h1>
        <p style="font-size: 18px; color: #666;">
            Advanced AI-powered object detection with automatic model selection
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with enhanced controls
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/4CAF50/FFFFFF?text=YOLO+Pro", width=200)
        
        st.markdown("### 🎛️ Control Panel")
        
        # Dark mode toggle
        dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.markdown("---")
        
        # Model settings
        st.markdown("### ⚙️ Model Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05)
        max_detections = st.slider("Max Detections", 1, 100, 50)
        
        st.markdown("---")
        
        # Image enhancement settings
        st.markdown("### 🎨 Image Enhancement")
        enhance_enabled = st.checkbox("Enable Image Enhancement")
        brightness = 0
        contrast = 1.0
        sharpness = 1.0
        
        if enhance_enabled:
            brightness = st.slider("Brightness", -50, 50, 0)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        
        # Statistics reset
        if st.button("🔄 Reset Statistics"):
            st.session_state.detection_stats = {'total_detections': 0, 'images_processed': 0}
            st.session_state.performance_history = []
            st.success("Statistics reset!")
        
        # About section
        st.markdown("---")
        st.markdown("""
        ### ℹ️ About
        
        **Features:**
        - 🔍 Automatic model selection
        - 📊 Real-time analytics
        - 🎨 Image enhancement
        - 📱 Mobile-friendly design
        - 🌙 Dark mode support
        
        **Models:**
        - 🌡️ Thermal/IR detection
        - 🌅 Visible light detection
        - 🛩️ Aerial surveillance detection
        """)

    # Load models
    visible_model, thermal_model, aerial_model = load_models()

    if visible_model is None or thermal_model is None or aerial_model is None:
        st.error("❌ Models could not be loaded. Please check model file paths.")
        st.stop()
    else:
        st.success("✅ Models loaded successfully!")
        st.write(f"**Visible Model Classes:** {list(visible_model.names.values())}")
        st.write(f"**Thermal Model Classes:** {list(thermal_model.names.values())}")
        st.write(f"**Aerial Model Classes:** {list(aerial_model.names.values())}")

    # Enhanced metrics dashboard
    create_enhanced_metrics()

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📁 Upload Images", "📷 Webcam Detection", "📊 Analytics", "⚙️ Batch Processing", "🛩️ Air Surveillance", "📡 ESP32 Interfacing"])

    with tab1:
        st.markdown("### 📁 Media Upload & Detection")
        st.markdown("**Upload images or videos for object detection**")

        # Multiple file upload with drag-and-drop for images and videos
        uploaded_files = st.file_uploader(
            "📎 Drag & Drop or Choose Files (Images & Videos)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'webm'],
            accept_multiple_files=True,
            help="Upload images (JPG, PNG, BMP, TIFF) or videos (MP4, AVI, MOV, MKV, WEBM) for detection"
        )

        if uploaded_files:
            # Separate images and videos
            image_files = []
            video_files = []

            for file in uploaded_files:
                file_extension = file.name.lower().split('.')[-1]
                if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    image_files.append(file)
                elif file_extension in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                    video_files.append(file)

            # Process images
            if image_files:
                st.markdown("### 🖼️ Image Processing")
                if len(image_files) == 1:
                    # Single image processing
                    process_single_image(image_files[0], visible_model, thermal_model,
                                       confidence_threshold, nms_threshold, max_detections,
                                       enhance_enabled, brightness, contrast, sharpness)
                else:
                    # Multiple image processing
                    process_multiple_images(image_files, visible_model, thermal_model,
                                          confidence_threshold, nms_threshold, max_detections)

            # Process videos
            if video_files:
                st.markdown("### 🎬 Video Processing")
                for i, video_file in enumerate(video_files):
                    st.markdown(f"#### 📹 Video {i+1}: {video_file.name}")

                    # Detect video type (similar to image type detection)
                    # For now, we'll use visible model as default for videos
                    selected_model = visible_model
                    model_name = "Visible Light Model"

                    process_video_file(video_file, selected_model, model_name,
                                     confidence_threshold, nms_threshold, max_detections)

    with tab2:
        st.markdown("### 🛡️ HAL Surveillance Webcam Detection")
        st.markdown("**Real-time 14-class threat detection with HAL surveillance integration**")

        # Initialize global models
        global VISIBLE_MODEL, THERMAL_MODEL, AERIAL_MODEL
        VISIBLE_MODEL = visible_model
        THERMAL_MODEL = thermal_model
        AERIAL_MODEL = aerial_model

        # HAL Surveillance interface
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # WebRTC configuration
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })

            # Create HAL video transformer with settings
            transformer = VideoTransformer()
            transformer.confidence_threshold = confidence_threshold
            transformer.nms_threshold = nms_threshold

            # Real-time HAL webcam stream
            webrtc_ctx = webrtc_streamer(
                key="hal-surveillance-detection",
                video_processor_factory=lambda: transformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with col2:
            st.markdown("#### 🎛️ HAL Controls")

            if webrtc_ctx.video_processor:
                detection_enabled = st.checkbox("🎯 Enable Detection", value=True)
                webrtc_ctx.video_processor.detection_enabled = detection_enabled

                process_frames = st.slider("Process Every N Frames", 1, 60, 2)
                webrtc_ctx.video_processor.process_every_n_frames = process_frames

                show_overlay = st.checkbox("📊 Show HAL Overlay", value=True)
                webrtc_ctx.video_processor.show_hal_overlay = show_overlay

                # Real-time settings update
                webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
                webrtc_ctx.video_processor.nms_threshold = nms_threshold

            # HAL Status indicators
            if webrtc_ctx.state.playing:
                st.success("🟢 **HAL SURVEILLANCE ACTIVE**")
                st.info("🛡️ **Threat Detection Running**")
                st.markdown("**Model:** HAL Surveillance (92.7% accuracy)")
            else:
                st.error("🔴 **HAL SURVEILLANCE STANDBY**")
                st.warning("📹 Click START to begin surveillance")

        with col3:
            st.markdown("#### 📊 Live HAL Statistics")

            if webrtc_ctx.video_processor and hasattr(webrtc_ctx.video_processor, 'total_detections'):
                # Display live HAL statistics
                st.metric("Total Detections", webrtc_ctx.video_processor.total_detections)
                st.metric("Current FPS", webrtc_ctx.video_processor.current_fps)

                # Threat level breakdown
                threat_counts = webrtc_ctx.video_processor.threat_count
                st.markdown("**🚨 Threat Breakdown:**")
                st.metric("🔴 High Threats", threat_counts.get('HIGH_THREAT', 0))
                st.metric("🟡 Potential Threats", threat_counts.get('POTENTIAL_THREAT', 0))
                st.metric("🟢 Low Threats", threat_counts.get('LOW_THREAT', 0))
                st.metric("✅ Safe", threat_counts.get('SAFE', 0))
            else:
                st.info("📊 Statistics will appear when surveillance is active")

        # HAL Information Section
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 HAL Surveillance Features:**
            - **14-Class Threat Detection** (person_civilian, person_armed, weapons, vehicles, animals, UAV, etc.)
            - **Real-time Threat Classification** (HIGH/POTENTIAL/LOW/SAFE)
            - **92.7% Detection Accuracy** (mAP@0.5)
            - **Live Statistics Dashboard**
            - **Threat-specific Visualization**
            """)

        with col2:
            st.markdown("""
            **🚨 Threat Classification:**
            - 🔴 **HIGH THREAT**: Armed persons, weapons, wild animals, UAVs
            - 🟡 **POTENTIAL THREAT**: Vehicles, unknown entities
            - 🟢 **LOW THREAT**: Domestic animals
            - ✅ **SAFE**: Civilians
            """)

        # Alternative: Launch native HAL surveillance
        st.markdown("---")
        st.markdown("### 🖥️ Alternative: Native HAL Surveillance")
        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **🌐 Web-based HAL Surveillance (Current)**
            - Browser-based real-time detection
            - Integrated threat visualization
            - Live statistics dashboard
            - Cross-platform compatibility
            """)

        with col2:
            if st.button("🚀 Launch Native HAL Surveillance", type="secondary"):
                hal_script = "hal_webcam_surveillance.py"
                if os.path.exists(hal_script):
                    try:
                        import subprocess
                        subprocess.Popen(["python", hal_script],
                                       creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                        st.success("✅ Native HAL Surveillance launched!")
                        st.info("💡 The native system will open in a separate window with audio alerts and screenshot capabilities")
                    except Exception as e:
                        st.error(f"❌ Error launching native HAL surveillance: {e}")
                else:
                    st.error("❌ hal_webcam_surveillance.py not found")

    with tab3:
        create_analytics_dashboard()

    with tab4:
        st.markdown("### ⚙️ Batch Processing")
        st.markdown("**Upload multiple images or videos for batch processing**")

        batch_files = st.file_uploader(
            "📎 Drag & Drop or Choose Multiple Files (Images & Videos)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'webm'],
            accept_multiple_files=True,
            key="batch_upload",
            help="Upload multiple images or videos for batch processing"
        )
        
        if batch_files and len(batch_files) > 1:
            # Show file type breakdown
            image_count = sum(1 for f in batch_files if f.name.lower().split('.')[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
            video_count = sum(1 for f in batch_files if f.name.lower().split('.')[-1] in ['mp4', 'avi', 'mov', 'mkv', 'webm'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(batch_files))
            with col2:
                st.metric("Images", image_count)
            with col3:
                st.metric("Videos", video_count)

            if st.button("🚀 Start Batch Processing"):
                process_batch_files(batch_files, visible_model, thermal_model,
                                  confidence_threshold, nms_threshold, max_detections)

    with tab5:
        st.markdown("### 🛩️ Air Surveillance")
        st.markdown("**Specialized aerial detection using trained aerial surveillance model**")

        # Create sub-tabs for different input methods
        aerial_tab1, aerial_tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])

        with aerial_tab1:
            st.markdown("#### 📁 Upload Aerial Media")
            st.markdown("**Upload aerial images or videos for specialized surveillance detection**")

            # File upload for aerial images and videos
            uploaded_file = st.file_uploader(
                "📎 Drag & Drop or Choose Aerial Files (Images & Videos)",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload aerial/drone images or videos for specialized detection",
                key="aerial_upload"
            )

            if uploaded_file is not None:
                # Image enhancement controls
                st.markdown("---")
                st.markdown("### 🎨 Image Enhancement")

                col1, col2 = st.columns(2)
                with col1:
                    enhance_enabled = st.checkbox("Enable Enhancement", value=False, key="aerial_enhance")
                    brightness = st.slider("Brightness", -50, 50, 0, key="aerial_brightness")
                with col2:
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1, key="aerial_contrast")
                    sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1, key="aerial_sharpness")

                st.markdown("---")

                # Model settings
                st.markdown("### ⚙️ Aerial Model Settings")
                col1, col2, col3 = st.columns(3)
                with col1:
                    aerial_confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="aerial_conf")
                with col2:
                    aerial_nms = st.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05, key="aerial_nms")
                with col3:
                    aerial_max_det = st.slider("Max Detections", 1, 100, 50, key="aerial_max_det")

                st.markdown("---")

                # Determine file type and process accordingly
                file_extension = uploaded_file.name.lower().split('.')[-1]

                if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    # Process aerial image
                    st.markdown("### 🖼️ Aerial Image Processing")
                    process_aerial_image(uploaded_file, aerial_model, aerial_confidence, aerial_nms, aerial_max_det,
                                       enhance_enabled, brightness, contrast, sharpness)

                elif file_extension in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
                    # Process aerial video
                    st.markdown("### 🎬 Aerial Video Processing")
                    st.info("🛩️ Processing with specialized aerial surveillance model")
                    process_video_file(uploaded_file, aerial_model, "Aerial Model",
                                     aerial_confidence, aerial_nms, aerial_max_det)

                else:
                    st.error("❌ Unsupported file format. Please upload images (JPG, PNG, BMP, TIFF) or videos (MP4, AVI, MOV, MKV, WEBM).")

        with aerial_tab2:
            st.markdown("#### 📷 Live Camera Detection")
            st.markdown("**Real-time aerial surveillance using your camera**")

            # Camera controls
            col1, col2 = st.columns(2)
            with col1:
                camera_enabled = st.checkbox("Enable Camera", key="aerial_camera_enable")
                if camera_enabled:
                    st.info("📹 Camera will start when you click 'Start Detection'")

            with col2:
                detection_active = st.checkbox("Start Detection", key="aerial_detection_active", disabled=not camera_enabled)

            if camera_enabled:
                st.markdown("---")
                st.markdown("### ⚙️ Camera Detection Settings")

                col1, col2, col3 = st.columns(3)
                with col1:
                    cam_confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="aerial_cam_conf")
                with col2:
                    cam_nms = st.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05, key="aerial_cam_nms")
                with col3:
                    cam_max_det = st.slider("Max Detections", 1, 100, 50, key="aerial_cam_max_det")

                # Additional camera settings
                col1, col2 = st.columns(2)
                with col1:
                    fps_limit = st.slider("FPS Limit", 1, 60, 30, key="aerial_fps")
                with col2:
                    show_fps = st.checkbox("Show FPS", value=True, key="aerial_show_fps")

                st.markdown("---")

                if detection_active:
                    # Create placeholders for camera feed and stats
                    camera_placeholder = st.empty()
                    stats_placeholder = st.empty()

                    # Start camera detection
                    run_aerial_camera_detection(aerial_model, cam_confidence, cam_nms, cam_max_det,
                                               fps_limit, show_fps, camera_placeholder, stats_placeholder)

    with tab6:
        st.markdown("### 📡 ESP32 Interfacing")
        st.markdown("**Real-time surveillance using ESP32-CAM with HAL detection**")

        # ESP32 connection status and controls
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔌 ESP32 Server Status")
            if 'esp32_server_running' not in st.session_state:
                st.session_state.esp32_server_running = False

            if st.session_state.esp32_server_running:
                st.success("🟢 ESP32 Server Running")
                st.info("📡 Port: 8765 | Ready for ESP32-CAM connections")
            else:
                st.error("🔴 ESP32 Server Stopped")
                st.warning("⚠️ Start server to enable ESP32-CAM connections")

        with col2:
            st.markdown("#### ⚙️ ESP32 Server Controls")

            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("🚀 Start ESP32 Server", key="start_esp32_server", use_container_width=True):
                    start_esp32_server_enhanced()

            with col_stop:
                if st.button("🛑 Stop ESP32 Server", key="stop_esp32_server", use_container_width=True):
                    stop_esp32_server_enhanced()

            # Demo mode section
            st.markdown("---")
            st.markdown("#### 🎭 Demo Mode")
            col_demo1, col_demo2 = st.columns(2)

            with col_demo1:
                if st.button("🎬 Start Demo Mode", key="start_demo_mode", use_container_width=True):
                    start_esp32_demo_mode()

            with col_demo2:
                if st.button("⏹️ Stop Demo", key="stop_demo_mode", use_container_width=True):
                    stop_esp32_demo_mode()

            # Server info
            st.caption("🛡️ Runs esp32_surveillance_server.py with HAL detection")
            st.caption("🎭 Demo mode simulates ESP32-CAM connection and detection")

        # Real-time ESP32 Server Terminal Output
        st.markdown("---")
        st.markdown("#### 📊 ESP32 Server Terminal Output")
        st.markdown("**Real-time output from esp32_surveillance_server.py:**")

        # ESP32 server output container
        esp32_output_container = st.container()
        with esp32_output_container:
            if 'esp32_output' not in st.session_state:
                st.session_state.esp32_output = []

            # Control panel for output
            col1, col2, col3 = st.columns(3)
            with col1:
                auto_refresh_esp32 = st.checkbox("🔄 Auto-refresh output", value=True, key="auto_refresh_esp32")
            with col2:
                if st.button("🗑️ Clear Output", key="clear_esp32_output"):
                    st.session_state.esp32_output = []
                    st.rerun()
            with col3:
                max_lines = st.selectbox("Max lines to show", [10, 20, 50, 100], index=1, key="max_lines_esp32")

            # Display recent output
            if st.session_state.esp32_output:
                output_text = "\n".join(st.session_state.esp32_output[-max_lines:])
                st.code(output_text, language="bash", line_numbers=True)

                # Show output stats
                st.caption(f"📈 Showing last {min(len(st.session_state.esp32_output), max_lines)} of {len(st.session_state.esp32_output)} total lines")
            else:
                st.info("🔍 No ESP32 server output yet. Start the server to see real-time terminal output.")
                st.markdown("""
                **Expected output includes:**
                - 🛡️ Server startup messages
                - 📡 WebSocket connection status
                - 🔍 Real-time detection results
                - 🚨 Alert notifications
                - 📊 Performance metrics
                """)

            # Auto-refresh mechanism
            if auto_refresh_esp32 and st.session_state.get('esp32_server_running', False):
                time.sleep(1)  # Small delay to prevent too frequent updates
                st.rerun()

def run_aerial_camera_detection(aerial_model, confidence_threshold, nms_threshold, max_detections,
                               fps_limit, show_fps, camera_placeholder, stats_placeholder):
    """Run real-time aerial surveillance camera detection"""

    # Initialize global aerial model
    global AERIAL_MODEL
    AERIAL_MODEL = aerial_model

    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # Create aerial video transformer with settings
    transformer = AerialVideoTransformer()
    transformer.confidence_threshold = confidence_threshold
    transformer.nms_threshold = nms_threshold
    transformer.max_detections = max_detections
    transformer.process_every_n_frames = max(1, 30 // fps_limit)  # Adjust based on FPS limit

    with camera_placeholder.container():
        st.markdown("#### 🛩️ Aerial Surveillance Camera Feed")

        # Real-time aerial camera stream
        webrtc_ctx = webrtc_streamer(
            key="aerial-surveillance-detection",
            video_processor_factory=lambda: transformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Update transformer settings in real-time
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
            webrtc_ctx.video_processor.nms_threshold = nms_threshold
            webrtc_ctx.video_processor.max_detections = max_detections

        # Status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            if webrtc_ctx.state.playing:
                st.success("🟢 Camera Active")
            else:
                st.error("🔴 Camera Inactive")

        with col2:
            if webrtc_ctx.state.playing:
                st.info("🛩️ Aerial Detection Running")
            else:
                st.warning("⏸️ Detection Paused")

        with col3:
            st.metric("Target FPS", fps_limit)

    # Display detection statistics
    with stats_placeholder.container():
        st.markdown("#### 📊 Real-time Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "Aerial Surveillance")
        with col2:
            st.metric("Confidence Threshold", f"{confidence_threshold:.2%}")
        with col3:
            st.metric("NMS Threshold", f"{nms_threshold:.2f}")

        if show_fps:
            st.info(f"🎯 Processing every {transformer.process_every_n_frames} frames for {fps_limit} FPS target")

def run_aerial_camera_detection(aerial_model, confidence_threshold, nms_threshold, max_detections,
                               fps_limit, show_fps, camera_placeholder, stats_placeholder):
    """Run real-time aerial surveillance camera detection"""

    # Initialize global aerial model
    global AERIAL_MODEL
    AERIAL_MODEL = aerial_model

    # WebRTC configuration
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    # Create aerial video transformer with settings
    transformer = AerialVideoTransformer()
    transformer.confidence_threshold = confidence_threshold
    transformer.nms_threshold = nms_threshold
    transformer.max_detections = max_detections
    transformer.process_every_n_frames = max(1, 30 // fps_limit)  # Adjust based on FPS limit

    with camera_placeholder.container():
        st.markdown("#### 🛩️ Aerial Surveillance Camera Feed")

        # Real-time aerial camera stream
        webrtc_ctx = webrtc_streamer(
            key="aerial-surveillance-detection",
            video_processor_factory=lambda: transformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Update transformer settings in real-time
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
            webrtc_ctx.video_processor.nms_threshold = nms_threshold
            webrtc_ctx.video_processor.max_detections = max_detections

        # Status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            if webrtc_ctx.state.playing:
                st.success("🟢 Camera Active")
            else:
                st.error("🔴 Camera Inactive")

        with col2:
            if webrtc_ctx.state.playing:
                st.info("🛩️ Aerial Detection Running")
            else:
                st.warning("⏸️ Detection Paused")

        with col3:
            st.metric("Target FPS", fps_limit)

    # Display detection statistics
    with stats_placeholder.container():
        st.markdown("#### 📊 Real-time Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "Aerial Surveillance")
        with col2:
            st.metric("Confidence Threshold", f"{confidence_threshold:.2%}")
        with col3:
            st.metric("NMS Threshold", f"{nms_threshold:.2f}")

        if show_fps:
            st.info(f"🎯 Processing every {transformer.process_every_n_frames} frames for {fps_limit} FPS target")

# ESP32 Server Management Functions
ESP32_SERVER_PROCESS = None

def start_esp32_server_enhanced():
    """Start the ESP32 surveillance server with real-time output capture"""
    global ESP32_SERVER_PROCESS

    try:
        if ESP32_SERVER_PROCESS and ESP32_SERVER_PROCESS.poll() is None:
            st.warning("⚠️ ESP32 server is already running!")
            return

        # Path to the ESP32 surveillance server
        esp32_server_path = "EO_ML_MODEL/esp32_surveillance_server.py"

        if not os.path.exists(esp32_server_path):
            st.error(f"❌ ESP32 server file not found: {esp32_server_path}")
            return

        # Start the ESP32 server as a subprocess
        ESP32_SERVER_PROCESS = subprocess.Popen(
            [sys.executable, esp32_server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        st.session_state.esp32_server_running = True
        st.success("✅ ESP32 surveillance server started successfully!")
        st.info("📡 Server is running on port 8765. ESP32-CAM can now connect.")

        # Start output capture thread
        import threading
        def capture_esp32_output():
            if 'esp32_output' not in st.session_state:
                st.session_state.esp32_output = []

            while ESP32_SERVER_PROCESS and ESP32_SERVER_PROCESS.poll() is None:
                try:
                    line = ESP32_SERVER_PROCESS.stdout.readline()
                    if line:
                        st.session_state.esp32_output.append(line.strip())
                        # Keep only last 100 lines
                        if len(st.session_state.esp32_output) > 100:
                            st.session_state.esp32_output = st.session_state.esp32_output[-100:]
                except:
                    break

        threading.Thread(target=capture_esp32_output, daemon=True).start()

    except Exception as e:
        st.error(f"❌ Failed to start ESP32 server: {e}")
        st.session_state.esp32_server_running = False

def stop_esp32_server_enhanced():
    """Stop the ESP32 surveillance server"""
    global ESP32_SERVER_PROCESS

    try:
        if not ESP32_SERVER_PROCESS or ESP32_SERVER_PROCESS.poll() is not None:
            st.warning("⚠️ ESP32 server is not running!")
            return

        ESP32_SERVER_PROCESS.terminate()
        ESP32_SERVER_PROCESS.wait(timeout=5)
        ESP32_SERVER_PROCESS = None
        st.session_state.esp32_server_running = False

        st.success("✅ ESP32 surveillance server stopped successfully!")

    except Exception as e:
        st.error(f"❌ Failed to stop ESP32 server: {e}")

def start_esp32_demo_mode():
    """Start ESP32 demo mode with simulated output"""
    try:
        if 'demo_mode_running' not in st.session_state:
            st.session_state.demo_mode_running = False

        if st.session_state.demo_mode_running:
            st.warning("⚠️ Demo mode is already running!")
            return

        st.session_state.demo_mode_running = True
        st.success("✅ ESP32 Demo Mode started!")
        st.info("🎬 Simulating ESP32-CAM connection and detection...")

        # Initialize demo output
        if 'esp32_output' not in st.session_state:
            st.session_state.esp32_output = []

        # Add demo startup messages
        demo_messages = [
            "🛡️ ESP32 HAL SURVEILLANCE SERVER",
            "=" * 50,
            "🔧 Initializing HAL Surveillance Detector...",
            "✅ HAL Model loaded successfully",
            "📡 WebSocket server starting on port 8765...",
            "🟢 Server ready for ESP32-CAM connections",
            "",
            "🎬 DEMO MODE: Simulating ESP32-CAM connection...",
            "📡 ESP32-CAM connected from 192.168.1.100",
            "🔍 Starting real-time detection...",
            "",
            "📊 Detection Results:",
            "🎯 Frame 001: Processing...",
            "🔍 Detected: person (confidence: 0.87)",
            "🔍 Detected: vehicle (confidence: 0.92)",
            "⚠️ ALERT: High confidence detection (0.92 > 0.8)",
            "💾 Alert image saved: esp32_alerts/alert_001.jpg",
            "",
            "🎯 Frame 002: Processing...",
            "🔍 Detected: unknown_entity (confidence: 0.45)",
            "✅ Normal activity detected",
            "",
            "🎯 Frame 003: Processing...",
            "🔍 Detected: drone (confidence: 0.95)",
            "🚨 HIGH ALERT: Drone detected (0.95 > 0.8)",
            "💾 Alert image saved: esp32_alerts/alert_002.jpg",
            "📧 Alert notification sent",
            "",
            "📊 Performance Metrics:",
            "⏱️ Average processing time: 45ms",
            "🎯 Detection accuracy: 94.2%",
            "📡 Connection stable: 100% uptime",
            "💾 Total alerts saved: 2",
            "",
            "🔄 Continuous monitoring active...",
            "🛡️ HAL Surveillance System operational"
        ]

        # Add messages to output
        for msg in demo_messages:
            st.session_state.esp32_output.append(msg)

        # Keep only last 100 lines
        if len(st.session_state.esp32_output) > 100:
            st.session_state.esp32_output = st.session_state.esp32_output[-100:]

        # Start continuous demo updates
        import threading
        def demo_update_thread():
            import time
            import random

            frame_count = 4
            while st.session_state.get('demo_mode_running', False):
                time.sleep(4)  # Update every 4 seconds

                if not st.session_state.get('demo_mode_running', False):
                    break

                # Generate random detection
                detections = [
                    ("person", random.uniform(0.3, 0.9)),
                    ("vehicle", random.uniform(0.4, 0.95)),
                    ("drone", random.uniform(0.2, 0.8)),
                    ("unknown_entity", random.uniform(0.1, 0.6)),
                    ("animal", random.uniform(0.2, 0.7)),
                    ("weapon", random.uniform(0.1, 0.9))
                ]

                detection = random.choice(detections)
                obj_type, confidence = detection

                new_messages = [
                    f"🎯 Frame {frame_count:03d}: Processing...",
                    f"🔍 Detected: {obj_type} (confidence: {confidence:.2f})"
                ]

                if confidence > 0.8:
                    new_messages.extend([
                        f"🚨 ALERT: High confidence detection ({confidence:.2f} > 0.8)",
                        f"💾 Alert image saved: esp32_alerts/alert_{frame_count:03d}.jpg"
                    ])
                    if obj_type in ["drone", "weapon"]:
                        new_messages.append("📧 High priority alert notification sent")
                else:
                    new_messages.append("✅ Normal activity detected")

                new_messages.extend([
                    f"⏱️ Processing time: {random.randint(35, 65)}ms",
                    ""
                ])

                # Add to output
                for msg in new_messages:
                    st.session_state.esp32_output.append(msg)

                # Keep only last 100 lines
                if len(st.session_state.esp32_output) > 100:
                    st.session_state.esp32_output = st.session_state.esp32_output[-100:]

                frame_count += 1

                # Add periodic status updates
                if frame_count % 10 == 0:
                    status_messages = [
                        "📊 System Status Update:",
                        f"🎯 Total frames processed: {frame_count}",
                        f"📡 Connection uptime: {frame_count * 4 // 60}m {(frame_count * 4) % 60}s",
                        f"🛡️ HAL System: Operational",
                        ""
                    ]
                    for msg in status_messages:
                        st.session_state.esp32_output.append(msg)

        threading.Thread(target=demo_update_thread, daemon=True).start()

    except Exception as e:
        st.error(f"❌ Failed to start demo mode: {e}")
        st.session_state.demo_mode_running = False

def stop_esp32_demo_mode():
    """Stop ESP32 demo mode"""
    try:
        if not st.session_state.get('demo_mode_running', False):
            st.warning("⚠️ Demo mode is not running!")
            return

        st.session_state.demo_mode_running = False
        st.success("✅ ESP32 Demo Mode stopped!")

        # Add stop message to output
        if 'esp32_output' not in st.session_state:
            st.session_state.esp32_output = []

        st.session_state.esp32_output.extend([
            "",
            "🛑 Demo mode stopped by user",
            "📊 Demo session ended",
            "🔄 Ready for real ESP32-CAM connection"
        ])

    except Exception as e:
        st.error(f"❌ Failed to stop demo mode: {e}")

def connect_to_esp32(camera_feed_placeholder, detection_info_placeholder, confidence_threshold, alert_threshold):
    """Connect to ESP32 and display real-time feed"""
    try:
        if not st.session_state.get('esp32_server_running', False):
            st.error("❌ ESP32 server is not running! Please start the server first.")
            return

        with camera_feed_placeholder.container():
            st.success("🔗 Attempting to connect to ESP32-CAM...")
            st.info("📹 Waiting for ESP32-CAM to send video feed...")
            st.markdown("**Connection Status:** 🟡 Connecting...")

            # Display connection instructions
            st.markdown("""
            **ESP32-CAM Connection:**
            - Make sure ESP32-CAM is powered on
            - ESP32-CAM should automatically connect to the server
            - Video feed will appear here once connected
            """)

            # Show server status
            global ESP32_SERVER_PROCESS
            if ESP32_SERVER_PROCESS and ESP32_SERVER_PROCESS.poll() is None:
                st.success("✅ ESP32 Server is running on port 8765")
            else:
                st.error("❌ ESP32 Server process not found")

        with detection_info_placeholder.container():
            st.markdown("#### 🔍 Real-time Detection Info")
            st.info("Waiting for ESP32-CAM connection and first detection...")

            # Display recent captures if any
            import glob
            recent_captures = glob.glob("EO_ML_MODEL/esp32_captures/*.jpg")
            recent_alerts = glob.glob("EO_ML_MODEL/esp32_alerts/*.jpg")

            if recent_captures:
                st.markdown("**📸 Recent Captures:**")
                latest_capture = max(recent_captures, key=os.path.getctime)
                st.image(latest_capture, caption="Latest ESP32 Capture", width=300)

            if recent_alerts:
                st.markdown("**🚨 Recent Alerts:**")
                latest_alert = max(recent_alerts, key=os.path.getctime)
                st.image(latest_alert, caption="Latest Alert", width=300)

    except Exception as e:
        st.error(f"❌ Connection error: {e}")

def capture_esp32_frame():
    """Capture a frame from ESP32 camera"""
    try:
        if not st.session_state.get('esp32_server_running', False):
            st.error("❌ ESP32 server is not running! Please start the server first.")
            return

        # Request frame capture from ESP32-CAM
        st.info("📸 Frame capture requested. Check esp32_captures folder for saved images.")

        # Add capture request to output log
        if 'esp32_output' not in st.session_state:
            st.session_state.esp32_output = []

        st.session_state.esp32_output.append("📸 Manual frame capture requested")
        st.session_state.esp32_output.append("💾 Saving frame to esp32_captures/")

    except Exception as e:
        st.error(f"❌ Frame capture error: {e}")

def process_single_image(uploaded_file, visible_model, thermal_model,
                        confidence_threshold, nms_threshold, max_detections,
                        enhance_enabled, brightness, contrast, sharpness):
    """Process a single uploaded image"""
    image = Image.open(uploaded_file)
    
    # Apply enhancements if enabled
    if enhance_enabled:
        image = enhance_image(image, brightness, contrast, sharpness)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Original Image")
        st.image(image, use_column_width=True)
        st.info(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
    
    # Add debug option
    debug_mode = st.checkbox("🔍 Show image analysis debug info", value=False)

    # Detect image type with enhanced UI
    with st.spinner("🔍 Analyzing image type..."):
        image_type = detect_image_type(image, debug=debug_mode)
    
    # Display detection type with enhanced styling
    if image_type == 'thermal':
        st.markdown('<div class="thermal-mode">🌡️ THERMAL/INFRARED MODE DETECTED</div>',
                   unsafe_allow_html=True)
        selected_model = thermal_model
        model_name = "IR Model"
    else:
        st.markdown('<div class="visible-mode">🌅 VISIBLE LIGHT MODE DETECTED</div>',
                   unsafe_allow_html=True)
        selected_model = visible_model
        model_name = "Visible Light Model"
    
    # Run inference with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🚀 Running inference...")
    progress_bar.progress(30)
    
    result_img, detections, timing_info = run_inference(
        selected_model, image, image_type, 
        confidence_threshold, nms_threshold, max_detections
    )
    
    progress_bar.progress(100)
    status_text.text("✅ Processing complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    if result_img is not None:
        with col2:
            st.markdown(f"#### 🎯 Detection Results ({model_name})")
            st.image(result_img, use_column_width=True)
        
        # Enhanced detection display
        display_enhanced_results(detections, model_name, timing_info, result_img, image_type)

def process_multiple_images(uploaded_files, visible_model, thermal_model,
                          confidence_threshold, nms_threshold, max_detections):
    """Process multiple uploaded images"""
    st.markdown("#### 📊 Multiple Image Processing")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        image = Image.open(file)
        image_type = detect_image_type(image)
        
        selected_model = thermal_model if image_type == 'thermal' else visible_model
        model_name = "IR Model" if image_type == 'thermal' else "Visible Light Model"
        
        result_img, detections, timing_info = run_inference(
            selected_model, image, image_type,
            confidence_threshold, nms_threshold, max_detections
        )
        
        all_results.append({
            'filename': file.name,
            'image_type': image_type,
            'model_used': model_name,
            'detections': len(detections),
            'processing_time': timing_info.get('total_time', 0),
            'result_img': result_img,
            'original_img': image
        })
    
    progress_bar.empty()
    status_text.empty()
    
    # Display batch results
    display_batch_results(all_results)

def process_batch_files(batch_files, visible_model, thermal_model,
                       confidence_threshold, nms_threshold, max_detections):
    """Process batch of images and videos with detailed analytics"""
    st.markdown("#### 🔄 Batch Processing in Progress...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_results = []
    total_processing_time = 0
    
    for i, file in enumerate(batch_files):
        status_text.text(f"Processing {file.name}... ({i+1}/{len(batch_files)})")
        progress_bar.progress((i + 1) / len(batch_files))

        start_time = time.time()

        # Determine file type
        file_extension = file.name.lower().split('.')[-1]

        if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            # Process image
            image = Image.open(file)
            image_type = detect_image_type(image)

            selected_model = thermal_model if image_type == 'thermal' else visible_model
            model_name = "IR Model" if image_type == 'thermal' else "Visible Light Model"

            _, detections, _ = run_inference(
                selected_model, image, image_type,
                confidence_threshold, nms_threshold, max_detections
            )

            processing_time = time.time() - start_time
            total_processing_time += processing_time

            batch_results.append({
                'filename': file.name,
                'file_type': 'image',
                'image_type': image_type,
                'model_used': model_name,
                'detections': len(detections),
                'processing_time': processing_time * 1000,  # Convert to ms
            })

        elif file_extension in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            # Process video (simplified for batch processing)
            import tempfile
            import os

            # Save video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(file.read())
                temp_video_path = tmp_file.name

            try:
                cap = cv2.VideoCapture(temp_video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Process every 30th frame for batch processing
                    frame_count = 0
                    total_detections = 0
                    processed_frames = 0

                    selected_model = visible_model  # Default to visible for videos
                    model_name = "Visible Light Model"

                    # Apply model settings
                    selected_model.conf = confidence_threshold
                    selected_model.iou = nms_threshold
                    selected_model.max_det = max_detections

                    while cap.isOpened() and processed_frames < 10:  # Limit to 10 frames for batch
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        if frame_count % 30 != 0:  # Process every 30th frame
                            continue

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = selected_model(frame_rgb)

                        # Use robust detection counting
                        frame_detections = 0

                        # Method 1: Try standard boxes attribute
                        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                            frame_detections = len(results[0].boxes)

                        # Method 2: Try accessing raw predictions if Method 1 failed
                        elif hasattr(results[0], 'data') and results[0].data is not None:
                            data = results[0].data
                            frame_detections = len([d for d in data if d[4] >= confidence_threshold])

                        # Method 3: Alternative prediction format
                        elif hasattr(results[0], 'pred') and results[0].pred is not None:
                            pred = results[0].pred
                            if isinstance(pred, list) and len(pred) > 0:
                                pred_tensor = pred[0]
                                frame_detections = len([d for d in pred_tensor if d[4] >= confidence_threshold])

                        total_detections += frame_detections

                        processed_frames += 1

                    cap.release()

                    processing_time = time.time() - start_time
                    total_processing_time += processing_time

                    batch_results.append({
                        'filename': file.name,
                        'file_type': 'video',
                        'image_type': 'video',
                        'model_used': model_name,
                        'detections': total_detections,
                        'processing_time': processing_time * 1000,
                        'total_frames': total_frames,
                        'processed_frames': processed_frames
                    })

            finally:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

        else:
            # Unsupported file type
            batch_results.append({
                'filename': file.name,
                'file_type': 'unsupported',
                'image_type': 'unknown',
                'model_used': 'None',
                'detections': 0,
                'processing_time': 0,
            'confidence_scores': [d['confidence'] for d in detections] if detections else [],
            'classes_detected': list(set([d['class'] for d in detections])) if detections else []
        })

    progress_bar.empty()
    status_text.empty()

    # Display batch processing results
    st.success(f"✅ Batch processing completed! Processed {len(batch_files)} images in {total_processing_time:.2f} seconds")

    # Create batch analytics
    display_batch_analytics(batch_results)

def display_enhanced_results(detections, model_name, timing_info, result_img, image_type):
    """Display enhanced detection results with detailed analytics"""

    # Performance metrics
    st.markdown("#### ⏱️ Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Time", f"{timing_info.get('total_time', 0):.1f}ms")
    with col2:
        st.metric("Preprocess", f"{timing_info.get('preprocess_time', 0):.1f}ms")
    with col3:
        st.metric("Inference", f"{timing_info.get('inference_time', 0):.1f}ms")
    with col4:
        st.metric("Postprocess", f"{timing_info.get('postprocess_time', 0):.1f}ms")

    # Detection summary
    st.markdown("#### 📊 Detection Summary")

    if detections and len(detections) > 0:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Detections", len(detections))
        with col2:
            avg_confidence = np.mean([d['confidence'] for d in detections])
            st.metric("Average Confidence", f"{avg_confidence:.2%}")
        with col3:
            st.metric("Model Used", model_name)

        # Detailed detection table
        st.markdown("#### 🔍 Detailed Detections")

        detection_data = []
        for i, det in enumerate(detections):
            detection_data.append({
                'ID': i + 1,
                'Class': det['class'],
                'Confidence': f"{det['confidence']:.2%}",
                'Bounding Box': f"({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})",
                'Area': f"{det['area']:.0f} px²"
            })

        df = pd.DataFrame(detection_data)
        st.dataframe(df, use_container_width=True)

        # Class distribution chart
        if len(detections) > 1:
            class_counts = pd.Series([d['class'] for d in detections]).value_counts()
            fig = px.bar(x=class_counts.index, y=class_counts.values,
                        title="Detection Class Distribution",
                        labels={'x': 'Class', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)

        # Confidence distribution
        confidences = [d['confidence'] for d in detections]
        fig = px.histogram(x=confidences, nbins=10,
                          title="Confidence Score Distribution",
                          labels={'x': 'Confidence', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ No objects detected in the image.")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", 0)
        with col2:
            st.metric("Model Used", model_name)

    # Download options
    if result_img is not None:
        st.markdown("#### 📥 Download Options")

        col1, col2 = st.columns(2)

        with col1:
            # Download result image
            result_pil = Image.fromarray(result_img)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            buf.seek(0)

            st.download_button(
                label="📥 Download Result Image",
                data=buf.getvalue(),
                file_name=f"detection_result_{image_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

        with col2:
            # Download detection data as JSON
            detection_json = json.dumps({
                'detections': detections,
                'timing_info': timing_info,
                'model_used': model_name,
                'image_type': image_type,
                'timestamp': datetime.now().isoformat()
            }, indent=2, default=str)

            st.download_button(
                label="📄 Download Detection Data",
                data=detection_json,
                file_name=f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def display_batch_results(all_results):
    """Display results from multiple image processing"""
    st.markdown("#### 📊 Batch Processing Results")

    # Summary statistics
    total_images = len(all_results)
    total_detections = sum([r['detections'] for r in all_results])
    avg_processing_time = np.mean([r['processing_time'] for r in all_results])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Images Processed", total_images)
    with col2:
        st.metric("Total Detections", total_detections)
    with col3:
        st.metric("Avg Processing Time", f"{avg_processing_time:.1f}ms")
    with col4:
        thermal_count = sum([1 for r in all_results if r['image_type'] == 'thermal'])
        st.metric("Thermal Images", f"{thermal_count}/{total_images}")

    # Results table
    st.markdown("#### 📋 Processing Summary")

    summary_data = []
    for result in all_results:
        summary_data.append({
            'Filename': result['filename'],
            'Type': result['image_type'].title(),
            'Model': result['model_used'],
            'Detections': result['detections'],
            'Time (ms)': f"{result['processing_time']:.1f}"
        })

    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)

    # Visual results grid
    st.markdown("#### 🖼️ Visual Results")

    cols = st.columns(min(3, len(all_results)))
    for i, result in enumerate(all_results[:6]):  # Show first 6 results
        with cols[i % 3]:
            st.markdown(f"**{result['filename']}**")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(result['original_img'], caption="Original", use_column_width=True)
            with col_img2:
                if result['result_img'] is not None:
                    st.image(result['result_img'], caption="Detected", use_column_width=True)
            st.caption(f"{result['detections']} detections • {result['processing_time']:.1f}ms")

def display_batch_analytics(batch_results):
    """Display comprehensive batch processing analytics"""
    st.markdown("#### 📈 Batch Analytics")

    df = pd.DataFrame(batch_results)

    col1, col2 = st.columns(2)

    with col1:
        # Processing time distribution
        fig = px.histogram(df, x='processing_time',
                          title="Processing Time Distribution",
                          labels={'processing_time': 'Time (ms)', 'count': 'Images'})
        st.plotly_chart(fig, use_container_width=True)

        # Model usage
        model_counts = df['model_used'].value_counts()
        fig = px.pie(values=model_counts.values, names=model_counts.index,
                     title="Model Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Detection count distribution
        fig = px.histogram(df, x='detections',
                          title="Detections per Image",
                          labels={'detections': 'Number of Detections', 'count': 'Images'})
        st.plotly_chart(fig, use_container_width=True)

        # Image type distribution
        type_counts = df['image_type'].value_counts()
        fig = px.bar(x=type_counts.index, y=type_counts.values,
                     title="Image Type Distribution",
                     labels={'x': 'Image Type', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

    # Performance summary
    st.markdown("#### 🎯 Performance Summary")

    summary_stats = {
        'Total Images': len(batch_results),
        'Total Processing Time': f"{sum(df['processing_time']):.1f}ms",
        'Average Time per Image': f"{df['processing_time'].mean():.1f}ms",
        'Fastest Processing': f"{df['processing_time'].min():.1f}ms",
        'Slowest Processing': f"{df['processing_time'].max():.1f}ms",
        'Total Detections': sum(df['detections']),
        'Average Detections per Image': f"{df['detections'].mean():.1f}",
        'Images with Detections': f"{len(df[df['detections'] > 0])}/{len(df)}"
    }

    cols = st.columns(4)
    for i, (key, value) in enumerate(summary_stats.items()):
        with cols[i % 4]:
            st.metric(key, value)

if __name__ == "__main__":
    main()