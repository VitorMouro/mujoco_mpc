// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/bicycle/bicycle.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc
{
  std::string Bicycle::XmlPath() const
  {
    return GetModelPath("bicycle/task.xml");
  }
  std::string Bicycle::Name() const { return "Bicycle"; }

  void Bicycle::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                     double *residual) const
  {

    int counter = 0;

    // Target speed for the goal heading
    double speed_goal = parameters_[0];
    double heading_goal = - parameters_[2]; // In radians [-pi, pi]

    double target_velocity[3] = {speed_goal * cos(heading_goal),
                                 speed_goal * sin(heading_goal), 0};
    double *currect_velocity = SensorByName(model, data, "frame_subtreelinvel");
    double velocity_error[3];  
    mju_sub3(velocity_error, target_velocity, currect_velocity);
    double velocity_error_norm = mju_norm3(velocity_error);
    residual[counter++] = velocity_error_norm;

    // Height target
    double height = SensorByName(model, data, "trace0")[2];
    residual[counter++] = height - parameters_[1];

    // ----- action ----- //
    mjtNum *start = data->ctrl + model->nu - 22;
    mju_copy(&residual[counter], start, 21);
    counter += 21;

    int user_sensor_dim = 0;
    for (int i = 0; i < model->nsensor; i++)
    {
      if (model->sensor_type[i] == mjSENS_USER)
      {
        user_sensor_dim += model->sensor_dim[i];
      }
    }
    if (user_sensor_dim != counter)
    {
      mju_error_i(
          "mismatch between total user-sensor dimension "
          "and actual length of residual %d",
          counter);
    }
  }

  constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};

  void Bicycle::ModifyScene(const mjModel *model, const mjData *data, mjvScene *scene) const
  {
    // Draw the goal heading
    mjtNum size[3] = {0.05, 0.05, residual_.parameters_[0]};
    mjtNum *pos = SensorByName(model, data, "bicycle_pos");
    double heading = residual_.parameters_[2];
    mjtNum mat_rotatedY[9] = {cos(mjPI/2), 0, sin(mjPI/2), 0, 1, 0, -sin(mjPI/2), 0, cos(mjPI/2)};
    mjtNum mat_rotatedX[9] = {1, 0, 0, 0, cos(heading), -sin(heading), 0, sin(heading), cos(heading)};
    mjtNum mat_res[9];
    mju_mulMatMat(mat_res, mat_rotatedY, mat_rotatedX, 3, 3, 3);
    AddGeom(scene, mjGEOM_ARROW, size, pos, mat_res, kStepRgba);
    // Draw the current heading
    // TODO
  }

  void Bicycle::TransitionLocked(mjModel *model, mjData *data)
  {
    // printf("Crank: %f\n", data->qvel[9]);
    // printf("Rear wheel: %f\n", data->qvel[8]);
    // if (!locked && data->qvel[9] >= data->qvel[8]) {
    //   data->eq_active[0] = 1;
    //   model->eq_data[0] = data->qpos[10] - data->qpos[9];
    //   locked = true;
    //   printf("Locked mode\n");
    // }
    // if (locked && data->qvel[9] < data->qvel[8]) {
    //   data->eq_active[0] = 0;
    //   locked = false;
    //   printf("Unlocked mode\n");
    // }
  }

} // namespace mjpc
