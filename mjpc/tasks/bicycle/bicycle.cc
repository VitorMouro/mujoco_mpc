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
    double heading_goal = parameters_[2]; // In radians [-pi, pi]
    double target_velocity[3] = {speed_goal * cos(heading_goal),
                                 speed_goal * sin(heading_goal), 0};
    double *currect_velocity = SensorByName(model, data, "frame_subtreelinvel");
    double velocity_error[3] = {target_velocity[0] - currect_velocity[0],
                                target_velocity[1] - currect_velocity[1],
                                target_velocity[2] - currect_velocity[2]};
    double velocity_error_norm = sqrt(velocity_error[0] * velocity_error[0] +
                                      velocity_error[1] * velocity_error[1] +
                                      velocity_error[2] * velocity_error[2]);
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

  void Bicycle::TransitionLocked(mjModel *model, mjData *data)
  {
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
