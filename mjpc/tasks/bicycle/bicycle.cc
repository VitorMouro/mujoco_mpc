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

namespace mjpc {
std::string Bicycle::XmlPath() const {
  return GetModelPath("bicycle/task.xml");
}
std::string Bicycle::Name() const { return "Bicycle"; }

void Bicycle::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {

  // Velocity target
  double x_vel = SensorByName(model, data, "frame_subtreelinvel")[0];
  double y_vel = SensorByName(model, data, "frame_subtreelinvel")[1];
  double resultant_vel = std::sqrt(x_vel * x_vel + y_vel * y_vel);
  residual[0] = resultant_vel - parameters_[0];

  // Height target
  double height = SensorByName(model, data, "trace0")[2];
  residual[1] = height - parameters_[1];

  /*residual[0] = std::cos(data->qpos[1]) - 1;
  residual[1] = data->qpos[0] - parameters_[0];
  residual[2] = data->qvel[1];
  residual[3] = data->ctrl[0];*/
}

}  // namespace mjpc
