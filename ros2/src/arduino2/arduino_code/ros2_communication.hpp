#pragma once
#include "motor_regulator.h"
extern Regulator left_regulator;
extern Regulator right_regulator;

enum Commands : uint8_t {
  SET_VELOCITY = 0x10,
  GET_ODOM_DATA = 0x11,
  HANDSHAKE_RESPONSE = 0x14
};

template<typename T>
void serial_read(T& value) {
  while(Serial.available() < static_cast<int>(sizeof(T))) {}
  Serial.readBytes(reinterpret_cast<uint8_t*>(&value), sizeof(T));
}

void set_velocity(float linear, float angular) {
  float left_speed = (linear - angular * WHEEL_BASE / 2) * TICKS_PER_METER;
  float right_speed = (linear + angular * WHEEL_BASE / 2) * TICKS_PER_METER;

  left_regulator.set_speed(left_speed);
  right_regulator.set_speed(right_speed);
}

struct EncoderData {
  int16_t left_ticks;
  int16_t right_ticks;
  float left_speed;
  float right_speed;
};

int16_t prev_left_ticks_ros = 0;
int16_t prev_right_ticks_ros = 0;

// Отправка данных энкодеров
void send_encoder_data() {
  // Мы не вызываем здесь calc_delta()! Он должен вызываться только в цикле 10 мс, 
  // иначе собьется расчет локальной скорости.
  
  long current_left = left_regulator.encoder.get_ticks_safe();
  long current_right = right_regulator.encoder.get_ticks_safe();

  int16_t left_delta = current_left - prev_left_ticks_ros;
  int16_t right_delta = current_right - prev_right_ticks_ros;

  prev_left_ticks_ros = current_left;
  prev_right_ticks_ros = current_right;

  // ИСПРАВЛЕНИЕ: encoder.speed - это тики за DT (0.01с). 
  // Чтобы получить тики/сек, нужно разделить на DT.
  float left_speed_mps = (left_regulator.encoder.speed / DT) * METERS_PER_TICK;
  float right_speed_mps = (right_regulator.encoder.speed / DT) * METERS_PER_TICK;

  EncoderData data{
    left_delta,
    right_delta,
    left_speed_mps,
    right_speed_mps
  };
  Serial.write((uint8_t*)&data, sizeof(data)); 
}

void handshake_response(){
  Serial.write("ARDUINO_OK");
  Serial.flush();
}

void command_spin() {
  if(Serial.available() > 0) {
    uint8_t cmd = Serial.read();
    switch(cmd) {
      case SET_VELOCITY:
        float linear, angular;
        serial_read(linear);
        serial_read(angular);
        set_velocity(linear, angular);
        break;
      case GET_ODOM_DATA: send_encoder_data(); break;
      case HANDSHAKE_RESPONSE: handshake_response(); break;
    }
  }
}
