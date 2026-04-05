#define MQ135_PIN 35

void setup() {
  Serial.begin(115200);
  analogSetPinAttenuation(MQ135_PIN, ADC_11db);
}

void loop() {
  int value = analogRead(MQ135_PIN);
  Serial.println(value);
  delay(1000);
}
