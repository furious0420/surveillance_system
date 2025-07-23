/*
 * ESP32 CAM + HAL Surveillance System
 * Real-time image capture and WebSocket communication
 *
 * Hardware: ESP32-CAM (AI-Thinker)
 * Libraries needed:
 * - ESP32 Arduino Core (install via Board Manager)
 * - ArduinoWebSockets by Markus Sattler (install via Library Manager)
 * - ArduinoJson by Benoit Blanchon (install via Library Manager)
 * - esp32-camera (included with ESP32 core)
 *
 * Author: HAL Surveillance Team
 * Version: 1.0
 * Date: 2024
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <base64.h>

// WiFi credentials - CHANGE THESE!
const char* ssid = "YOUR_WIFI_SSID";           // Replace with your WiFi name
const char* password = "YOUR_WIFI_PASSWORD";   // Replace with your WiFi password

// WebSocket server details - CHANGE THIS!
const char* websocket_server = "192.168.1.100";  // Replace with your computer's IP address
const int websocket_port = 8765;

// Camera configuration for AI-Thinker ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Global variables
WebSocketsClient webSocket;
bool connected = false;
unsigned long lastCaptureTime = 0;
const unsigned long captureInterval = 2000; // Capture every 2 seconds
bool alertMode = false;
unsigned long alertStartTime = 0;
const unsigned long alertDuration = 10000; // Alert mode for 10 seconds

// LED pins
#define LED_BUILTIN 4
#define FLASH_LED 4

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("🛡️ ESP32 HAL SURVEILLANCE CAMERA");
  Serial.println("================================");
  
  // Initialize LED
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  // Initialize camera
  if (initCamera()) {
    Serial.println("✅ Camera initialized successfully");
  } else {
    Serial.println("❌ Camera initialization failed");
    ESP.restart();
  }
  
  // Connect to WiFi
  connectToWiFi();
  
  // Initialize WebSocket
  initWebSocket();
  
  Serial.println("🚀 ESP32 HAL Surveillance ready!");
  Serial.println("📡 Connecting to surveillance server...");
}

void loop() {
  webSocket.loop();
  
  // Check if it's time to capture and send image
  if (connected && (millis() - lastCaptureTime > captureInterval)) {
    captureAndSendImage();
    lastCaptureTime = millis();
  }
  
  // Handle alert mode
  if (alertMode && (millis() - alertStartTime > alertDuration)) {
    alertMode = false;
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("🔕 Alert mode ended");
  }
  
  delay(100);
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Image quality settings
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA; // 1600x1200
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA; // 800x600
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x", err);
    return false;
  }
  
  // Adjust camera settings for better detection
  sensor_t * s = esp_camera_sensor_get();
  s->set_brightness(s, 0);     // -2 to 2
  s->set_contrast(s, 0);       // -2 to 2
  s->set_saturation(s, 0);     // -2 to 2
  s->set_special_effect(s, 0); // 0 to 6 (0-No Effect, 1-Negative, 2-Grayscale, 3-Red Tint, 4-Green Tint, 5-Blue Tint, 6-Sepia)
  s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
  s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
  s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
  s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
  s->set_aec2(s, 0);           // 0 = disable , 1 = enable
  s->set_ae_level(s, 0);       // -2 to 2
  s->set_aec_value(s, 300);    // 0 to 1200
  s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
  s->set_agc_gain(s, 0);       // 0 to 30
  s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
  s->set_bpc(s, 0);            // 0 = disable , 1 = enable
  s->set_wpc(s, 1);            // 0 = disable , 1 = enable
  s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
  s->set_lenc(s, 1);           // 0 = disable , 1 = enable
  s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
  s->set_vflip(s, 0);          // 0 = disable , 1 = enable
  s->set_dcw(s, 1);            // 0 = disable , 1 = enable
  s->set_colorbar(s, 0);       // 0 = disable , 1 = enable
  
  return true;
}

void connectToWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("📶 Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("✅ WiFi connected!");
  Serial.print("📍 IP address: ");
  Serial.println(WiFi.localIP());
}

void initWebSocket() {
  webSocket.begin(websocket_server, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  Serial.println("🔌 WebSocket client initialized");
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("❌ WebSocket Disconnected");
      connected = false;
      digitalWrite(LED_BUILTIN, LOW);
      break;
      
    case WStype_CONNECTED:
      Serial.printf("✅ WebSocket Connected to: %s\n", payload);
      connected = true;
      digitalWrite(LED_BUILTIN, HIGH);
      
      // Send ping to test connection
      sendPing();
      break;
      
    case WStype_TEXT:
      handleServerMessage((char*)payload);
      break;
      
    case WStype_ERROR:
      Serial.printf("❌ WebSocket Error: %s\n", payload);
      break;
      
    default:
      break;
  }
}

void handleServerMessage(const char* message) {
  DynamicJsonDocument doc(2048);
  deserializeJson(doc, message);
  
  String type = doc["type"];
  
  if (type == "welcome") {
    Serial.println("🛡️ Connected to HAL Surveillance Server");
    Serial.println("📡 Real-time detection active");
  }
  else if (type == "detection_result") {
    JsonObject result = doc["result"];
    
    String primaryClass = result["primary_class"];
    float confidence = result["primary_confidence"];
    String threatLevel = result["threat_level"];
    bool isAlert = result["alert"];
    
    Serial.printf("🔍 Detection: %s (%.1f%%) - %s\n", 
                  primaryClass.c_str(), confidence * 100, threatLevel.c_str());
    
    if (isAlert) {
      Serial.println("🚨 ALERT DETECTED!");
      triggerAlert();
    }
  }
  else if (type == "pong") {
    Serial.println("🏓 Pong received");
  }
  else if (type == "error") {
    String errorMsg = doc["message"];
    Serial.printf("❌ Server error: %s\n", errorMsg.c_str());
  }
}

void captureAndSendImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Camera capture failed");
    return;
  }
  
  // Convert image to base64
  String base64Image = base64::encode(fb->buf, fb->len);
  
  // Create JSON message
  DynamicJsonDocument doc(base64Image.length() + 200);
  doc["type"] = "image";
  doc["image"] = base64Image;
  doc["timestamp"] = millis();
  doc["width"] = fb->width;
  doc["height"] = fb->height;
  doc["format"] = "jpeg";
  
  String message;
  serializeJson(doc, message);
  
  // Send to server
  webSocket.sendTXT(message);
  
  Serial.printf("📸 Image sent (%dx%d, %d bytes)\n", fb->width, fb->height, fb->len);
  
  esp_camera_fb_return(fb);
}

void sendPing() {
  DynamicJsonDocument doc(128);
  doc["type"] = "ping";
  doc["timestamp"] = millis();
  
  String message;
  serializeJson(doc, message);
  webSocket.sendTXT(message);
}

void triggerAlert() {
  alertMode = true;
  alertStartTime = millis();
  
  // Flash LED rapidly
  for (int i = 0; i < 10; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(100);
  }
  
  digitalWrite(LED_BUILTIN, HIGH); // Keep LED on during alert
}
