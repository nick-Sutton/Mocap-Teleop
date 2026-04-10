#ifndef LIE_TLM_H
#define LIE_TLM_H

typedef struct {
  double q_err[NUM_U];

} LieTelemetry;

typedef struct {
  TelemetryAttribute q_err[NUM_U];

} TelemetryAttributes;

#endif //LIE_TLM_H
