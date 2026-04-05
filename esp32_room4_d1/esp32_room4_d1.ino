/**
 * Room 4 hardware node D1 — DHT11 + MQ-135 → POST /api/twin/reading
 *
 * Pins: DHT11 DATA GPIO4; MQ-135 AOUT GPIO34 (ADC1, OK with WiFi on).
 * Libs: DHT + Adafruit Unified Sensor, ArduinoJson v6+.
 *
 * API host: use the IPv4 of the adapter that shares your WiFi LAN with the ESP32.
 *   ipconfig → "Wireless LAN adapter Wi-Fi" → IPv4 (often 192.168.x.x).
 * Use "Wireless LAN adapter Wi-Fi" IPv4 from ipconfig (e.g. 172.20.10.x on a phone hotspot).
 * Do not use vEthernet/WSL (e.g. 172.27.112.1) — that is a different virtual network.
 * Run: .\run_api_lan.ps1  (Capstone folder). If POST -1: open_firewall_port8000.ps1 as Admin.
 * Hotspots: PC and ESP must be on same SSID; some routers block client-to-client (AP isolation).
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

const char *WIFI_SSID = "Lakshya";
const char *WIFI_PASSWORD = "12345678";

// PC "Wireless LAN adapter Wi-Fi" IPv4 from ipconfig (yours: 172.20.10.4; gateway 172.20.10.1).
const char *TWIN_POST_URL = "http://172.20.10.4:8000/api/twin/reading";

#define DHT_PIN 4
#define DHT_TYPE DHT11
#define MQ135_PIN 34

DHT dht(DHT_PIN, DHT_TYPE);

const unsigned long READ_INTERVAL_MS = 5000;
const unsigned long WIFI_RETRY_MS = 30000;

unsigned long lastRead = 0;
unsigned long lastWifiAttempt = 0;

static float mapf(float x, float inMin, float inMax, float outMin, float outMax) {
  if (inMax <= inMin) return outMin;
  x = constrain(x, inMin, inMax);
  return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

float mq135ToAqi(int raw) {
  raw = constrain(raw, 300, 4000);
  return mapf((float)raw, 300.0f, 4000.0f, 35.0f, 260.0f);
}

void connectWifi() {
  if (WiFi.status() == WL_CONNECTED) return;

  unsigned long now = millis();
  if (now - lastWifiAttempt < WIFI_RETRY_MS && lastWifiAttempt != 0) return;
  lastWifiAttempt = now;

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting WiFi");
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print(".");
    tries++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Gateway: ");
    Serial.println(WiFi.gatewayIP());
    Serial.println("Set TWIN_POST_URL to PC Wi-Fi IPv4 (same subnet as gateway) if POST fails.");
  } else {
    Serial.println("WiFi failed (will retry)");
  }
}

bool postReading(float aqi, float temperature, float humidity, float voltage) {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  http.setTimeout(20000);
  if (!http.begin(TWIN_POST_URL)) {
    Serial.println("http.begin(URL) failed — check TWIN_POST_URL");
    return false;
  }
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<256> doc;
  doc["aqi"] = aqi;
  doc["temperature"] = temperature;
  doc["humidity"] = humidity;
  doc["voltage"] = voltage;

  char body[192];
  size_t n = serializeJson(doc, body, sizeof(body));
  if (n == 0 || n >= sizeof(body)) {
    http.end();
    return false;
  }

  int code = http.POST(body);
  if (code < 0) {
    Serial.printf("POST %d (%s)\n", code, HTTPClient::errorToString(code).c_str());
  } else {
    Serial.printf("POST %d\n", code);
    if (code > 0) Serial.println(http.getString());
  }
  http.end();
  return code >= 200 && code < 300;
}

void setup() {
  Serial.begin(115200);
  delay(800);
  dht.begin();
  analogSetAttenuation(ADC_11db);
  connectWifi();
}

void loop() {
  unsigned long now = millis();
  if (now - lastRead < READ_INTERVAL_MS) {
    delay(50);
    return;
  }
  lastRead = now;

  connectWifi();

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("DHT read failed");
    return;
  }

  int mqRaw = analogRead(MQ135_PIN);
  float aqi = mq135ToAqi(mqRaw);
  const float voltage = 220.0f;

  Serial.printf("T=%.1f C H=%.0f%% MQ=%d AQI~=%.0f\n", temperature, humidity, mqRaw, aqi);

  if (!postReading(aqi, temperature, humidity, voltage)) {
    Serial.println("POST failed");
  }
}
