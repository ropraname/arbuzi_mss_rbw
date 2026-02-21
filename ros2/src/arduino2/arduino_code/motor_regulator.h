#pragma once
#include "robot_params.h"

struct PID {
  float kp, ki, kd, max_i;
  float integral = 0;
  float prev_error = 0;

  PID(float p, float i, float d, float mi) 
    : kp(p), ki(i), kd(d), max_i(mi) {}

  int calc(float error) {
    integral += error * DT * ki;
    integral = constrain(integral, -max_i, max_i);
    float diff = (error - prev_error) * kd / DT;
    prev_error = error;
    return static_cast<int>(error * kp + integral + diff);
  }
};

struct Encoder {
  volatile long ticks = 0;
  long prev_ticks = 0;
  int speed = 0;
  byte pin_a, pin_b;
  bool invert;

  Encoder(byte a, byte b, void(*isr)(), bool inv) 
    : pin_a(a), pin_b(b), invert(inv) {
    pinMode(pin_a, INPUT);
    pinMode(pin_b, INPUT);
    attachInterrupt(digitalPinToInterrupt(pin_a), isr, RISING);
  }

  void encoder_int() {
    int dir = (digitalRead(pin_b) ^ invert) ? 1 : -1;
    ticks += dir;
  }

  // НОВОЕ: Безопасное чтение 32-битной переменной (защита от разрыва байтов прерыванием)
  long get_ticks_safe() {
    noInterrupts();
    long t = ticks;
    interrupts();
    return t;
  }

  void calc_delta() {
    long current_ticks = get_ticks_safe();
    speed = current_ticks - prev_ticks;
    prev_ticks = current_ticks;
  }
};

struct Motor {
  byte pin_dir, pin_pwm;
  bool invert;
  
  Motor(byte dir, byte pwm, bool inv = false) : pin_dir(dir), pin_pwm(pwm), invert(inv) {
    pinMode(pin_dir, OUTPUT);
    pinMode(pin_pwm, OUTPUT);
  }

  void set_pwmdir(int speed) {
    speed = constrain(speed, -255, 255);
    if (invert) speed = -speed;
    digitalWrite(pin_dir, speed > 0);
    analogWrite(pin_pwm, abs(speed));
  }
};

struct Regulator {
  Motor& motor;
  Encoder& encoder;
  PID& pid;
  
  float target_speed = 0;
  float current_speed = 0;
  double position_target = 0;
  float max_accel;
  
  unsigned long last_nonzero_time = 0;
  bool position_reset = false;

  Regulator(Motor& m, Encoder& e, PID& p)
    : motor(m), encoder(e), pid(p) {
    max_accel = MAX_LIN_ACCEL * TICKS_PER_METER;
  }

  void set_speed(float new_speed) {
    if ((target_speed > 0 && new_speed < 0) || (target_speed < 0 && new_speed > 0)) {
      position_target = encoder.get_ticks_safe();
      pid.integral = 0;
    }
    
    if (target_speed == 0 && new_speed != 0) {
      position_target = encoder.get_ticks_safe();
      position_reset = false;
    }
    
    if (new_speed != 0) {
      last_nonzero_time = millis();
    }
    
    target_speed = constrain(new_speed, -MAX_DELTA_TICKS, MAX_DELTA_TICKS);
  }

  // НОВОЕ: Принимаем множитель синхронизации (от 0.0 до 1.0)
  void update(float sync_mult = 1.0f) {
    long current_ticks = encoder.get_ticks_safe();

    // Плавное изменение скорости масштабируется общим коэффициентом синхронизации
    float speed_diff = target_speed - current_speed;
    float allowed_diff = copysignf(max_accel * DT * sync_mult, speed_diff);
    
    current_speed += (fabs(speed_diff) > fabs(allowed_diff)) ? allowed_diff : speed_diff;
    
    if (current_speed == 0) {
      position_target = current_ticks;
    } 
    else {
      // И сама цель тоже продвигается с учетом коэффициента синхронизации
      position_target += current_speed * DT * sync_mult;
    }
    
    if (target_speed == 0 && current_speed == 0) {
      if (millis() - last_nonzero_time > 1000 && !position_reset) {
        position_target = current_ticks;
        pid.integral = 0;
        position_reset = true;
      }
    }
    
    // Расчет PWM
    int error = position_target - current_ticks;
    int pwm = position_reset ? 0 : pid.calc(error);
    motor.set_pwmdir(pwm);
    
    encoder.calc_delta();
  }
};
