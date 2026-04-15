#ifndef WALK_IDQP_TLM_H
#define WALK_IDQP_TLM_H

typedef struct {
  int swing_leg[4];
  double swing_feet_err[12];

} WalkIdqpTelemetry;

typedef struct {
  TelemetryAttribute swing_leg[4];
  TelemetryAttribute swing_feet_err[12];

} TelemetryAttributes;

#endif //WALK_IDQP_TLM_H

