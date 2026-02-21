#include "motor_regulator.h"
#include "ros2_communication.hpp"

// Форвард-декларации
class Regulator;
extern Regulator left_regulator;
extern Regulator right_regulator;

Motor right_motor(4, 5, false);
Motor left_motor(7, 6, true);

// Создаем энкодеры с корректными пинами и настройками
Encoder left_enc(3, 12, []{ left_regulator.encoder.encoder_int(); }, false);
// INT0 (pin 2), B=11, invert=true
Encoder right_enc(2, 13, []{ right_regulator.encoder.encoder_int(); }, true);
// INT1 (pin 3), B=12, invert=false

// Создаем PID-регуляторы
PID left_pid(4.0, 0.2, 0.04, 100);
PID right_pid(4.0, 0.2, 0.04, 100);

// Создаем регуляторы, передавая созданные объекты
Regulator left_regulator(left_motor, left_enc, left_pid);
Regulator right_regulator(right_motor, right_enc, right_pid);

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  set_velocity(0.0, 0.0);
}

void loop() {
  static uint32_t t = millis();
  command_spin();
  
  if(millis() - t >= 10) { // DT = 0.01 сек
    t = millis();

    // 1. Безопасное чтение текущих позиций
    long left_pos = left_enc.get_ticks_safe();
    long right_pos = right_enc.get_ticks_safe();

    // 2. Расчет ошибки следования (насколько физика отстает от математики)
    long left_err = abs(left_regulator.position_target - left_pos);
    long right_err = abs(right_regulator.position_target - right_pos);
    long max_err = max(left_err, right_err);

    // 3. Вычисление коэффициента синхронизации
    float sync_mult = 1.0f;
    // Если любой мотор отстал сильнее допустимого, мы "замедляем время" генерации профиля
    if (max_err > MAX_TRACKING_ERROR) {
      sync_mult = (float)MAX_TRACKING_ERROR / max_err;
    }

    // 4. Обновление обоих регуляторов с одинаковым (общим!) множителем
    left_regulator.update(sync_mult);
    right_regulator.update(sync_mult);
  }
//  Serial.print(left_enc.ticks);
//  Serial.print("   ");
//  Serial.println(right_enc.ticks);
  
}
