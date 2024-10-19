#include "mjpc/tasks/bicycle/bicycle.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "GLFW/glfw3.h"

namespace mjpc
{
    std::string Bicycle::XmlPath() const
    {
        return GetModelPath("bicycle/task.xml");
    }
    std::string Bicycle::Name() const { return "Bicycle"; }

    Bicycle::Bicycle() : residual_(this)
    {
        path_ = new Path(20);

        double p0[9] = { 0.0000, 0.0000, 1.0000,
        -1.0000, 0.0000, 1.0000,
        1.0000, 0.0000, 1.0000 };

        double p1[9] = { 5.0000, 0.0000, 1.0000,
        4.0000, 0.0000, 1.0000,
        6.0000, 0.0000, 1.0000 };

        double p2[9] = { 10.0000, 0.0000, 1.0000,
        7.0000, 0.0000, 1.0000,
        13.0000, 0.0000, 1.0000 };

        double p3[9] = { 15.0000, -5.0000, 1.0000,
        15.0000, -2.0000, 1.0000,
        15.0000, -8.0000, 1.0000 };

        double p4[9] = { 15.0000, -15.0000, 1.0000,
        15.0000, -12.0000, 1.0000,
        15.0000, -18.0000, 1.0000 };

        double p5[9] = { 25.0000, -20.0000, 1.0000,
        22.2811, -21.2679, 1.0000,
        27.7189, -18.7321, 1.0000 };

        path_->addPoint(p0);
        path_->addPoint(p1);
        path_->addPoint(p2);
        path_->addPoint(p3);
        path_->addPoint(p4);
        path_->addPoint(p5);
    }

    Bicycle::~Bicycle()
    {
        delete path_;
    }

    int GetVelocityGoal(mjtNum *vel, mjtNum *head)
    {
        int count;
        const float *axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &count);
        if (count < 5)
            return 0;

        float heading = -axes[0] * M_PI;
        if (head != nullptr)
            *head = heading;
        float max_speed = 5;
        float speed = (axes[4] + 1) / 2 * max_speed;

        if (vel != nullptr)
        {
            vel[0] = speed * cos(heading);
            vel[1] = speed * sin(heading);
            vel[2] = 0;
        }
        return 1;
    }

    void ActionResidual(const mjModel *model, const mjData *data, double *residual, int *counter)
    {
        int humanoid_controls = 21;
        mjtNum *start = data->ctrl + model->nu - humanoid_controls;
        mju_copy(&residual[*counter], start, humanoid_controls);
        *counter += humanoid_controls;
    }

    void PoseResidual(const mjModel *model, const mjData *data, double *residual, int *counter)
    {
        // Arms are at last 6 positions of qpos
        mjtNum arms[6] = {0.477525, -0.31974, -0.750274, 0.477525, -0.31974, -0.750274};
        mjtNum arms_error[6];
        mju_sub(arms_error, data->qpos + model->nq - 6, arms, 6);
        mju_copy(&residual[*counter], arms_error, 6);
        *counter += 6;

        // Abdomen is at last model-nq - 21 to model-nq - 18
        mjtNum abdomen[3] = {0.0, -0.26, 0.0};
        mjtNum abdomen_error[3];
        mju_sub(abdomen_error, data->qpos + model->nq - 21, abdomen, 3);
        mju_copy(&residual[*counter], abdomen_error, 3);
        *counter += 3;
    }

    void VelocityResidual(const mjModel *model, const mjData *data, double *residual, int *counter, const std::vector<double> &parameters_)
    {
        double speed_goal = parameters_[0];
        double heading_goal = -parameters_[1]; // In radians [-pi, pi]

        double target_velocity[3] = {speed_goal * cos(heading_goal),
                                     speed_goal * sin(heading_goal), 0};

        GetVelocityGoal(target_velocity, nullptr);

        double *currect_velocity = SensorByName(model, data, "frame_subtreelinvel");
        double velocity_error[3];
        mju_sub3(velocity_error, target_velocity, currect_velocity);
        double velocity_error_norm = mju_norm3(velocity_error);
        residual[(*counter)++] = velocity_error_norm;
    }

    void BalanceResidual(const mjModel *model, const mjData *data, double *residual, int *counter)
    {
        mjtNum *up_axis = SensorByName(model, data, "bicycle_yaxis");
        residual[(*counter)++] = up_axis[2] - 1.0;
    }

    void PositionResidual(const mjModel *model, const mjData *data, double *residual, int *counter)
    {
        mjtNum *goal_pos = SensorByName(model, data, "goal_pos");
        mjtNum *bicycle_pos = SensorByName(model, data, "bicycle_pos");
        mjtNum goal_displacement[3];
        mju_sub3(goal_displacement, goal_pos, bicycle_pos);
        mjtNum goal_distance = mju_norm3(goal_displacement);
        residual[(*counter)++] = goal_distance;
    }

    void GoalResidual(const mjModel *model, const mjData *data, double *residual, int *counter, const std::vector<double> &parameters_)
    {
        // The bicycle should reach the goal position at a certain speed and heading
        mjtNum *goal_pos = SensorByName(model, data, "goal_pos");
        mjtNum *bicycle_pos = SensorByName(model, data, "bicycle_pos");

        mjtNum goal_displacement[3];
        mju_sub3(goal_displacement, goal_pos, bicycle_pos);
        goal_displacement[2] = 0; // Ignore the z-axis
        mjtNum goal_distance = mju_norm3(goal_displacement);
        residual[(*counter)++] = goal_distance;

        mjtNum goal_speed = parameters_[0];
        mjtNum *goal_xaxis = SensorByName(model, data, "goal_zaxis");
        mjtNum goal_velocity[3];
        mju_scl3(goal_velocity, goal_xaxis, goal_speed);
        mjtNum *bicycle_velocity = SensorByName(model, data, "frame_subtreelinvel");
        mjtNum velocity_error[3];
        mju_sub3(velocity_error, goal_velocity, bicycle_velocity);
        residual[(*counter)++] = mju_norm3(velocity_error);
    }

    void PathResidual(const mjModel *model, const mjData *data, double *residual, int *counter, const std::vector<double> &parameters_) {

    }

    void Bicycle::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const
    {
        int counter = 0;

        PositionResidual(model, data, residual, &counter);
        VelocityResidual(model, data, residual, &counter, parameters_);
        BalanceResidual(model, data, residual, &counter);
        ActionResidual(model, data, residual, &counter);
        PoseResidual(model, data, residual, &counter);
        GoalResidual(model, data, residual, &counter, parameters_);

        int user_sensor_dim = 0;
        for (int i = 0; i < model->nsensor; i++)
            if (model->sensor_type[i] == mjSENS_USER)
                user_sensor_dim += model->sensor_dim[i];
        if (user_sensor_dim != counter)
            mju_error_i(
                "mismatch between total user-sensor dimension "
                "and actual length of residual %d",
                counter);
    }

    void Bicycle::ModifyScene(const mjModel *model, const mjData *data, mjvScene *scene) const
    {
        float segment_color[4] = {0.0, 0.0, 0.0, 0.4};
        double zero3[3] = {0};
        double zero9[9] = {0};
        float width = 0.02;
        int res = 100;
        int n_anchors = path_->getNumAnchors();

        for (size_t i = 0; i < res; i++)
        {
            // check max geoms
            if (scene->ngeom >= scene->maxgeom)
            {
                printf("max geom!!!\n");
                continue;
            }

            // initialize geometry
            mjv_initGeom(&scene->geoms[scene->ngeom], mjGEOM_CAPSULE, zero3, zero3, zero9,
                         segment_color);

            // make geometry
            double a[3], b[3];
            double t0 = (double)i/res * n_anchors;
            double t1 = (double)(i+1)/res * n_anchors;
            path_->getPoint(a, t0);
            path_->getPoint(b, t1);

            mjv_makeConnector(
                &scene->geoms[scene->ngeom], mjGEOM_CAPSULE, width,
                a[0], a[1], a[2], b[0], b[1], b[2]);
            // increment number of geometries
            scene->ngeom += 1;
        }

        // Draw the anchors
        float anchor_color[4] = {0.0, 0.0, 1.0, 1.0};
        for (int i = 0; i < n_anchors; i++)
        {
            double pos[3];
            path_->getAnchor(pos, i);
            double size[3] = {0.1, 0.1, 0.1};
            AddGeom(scene, mjGEOM_SPHERE, size, pos, nullptr, anchor_color);
        }

        // Draw the controls
        // float control_color [4] = {1.0, 0.0, 0.0, 1.0};
        // for (int i = 0; i < n_anchors; i++)
        // {
        //     double a[3], b[3];
        //     path_->getLeftControl(a, i);
        //     path_->getRightControl(b, i);
        //     double size[3] = {0.05, 0.05, 0.05};
        //     AddGeom(scene, mjGEOM_SPHERE, size, a, nullptr, control_color);
        //     AddGeom(scene, mjGEOM_SPHERE, size, b, nullptr, control_color);
        // }
    }

    void Bicycle::TransitionLocked(mjModel *model, mjData *data)
    {
        mjtNum *current_pos = SensorByName(model, data, "bicycle_pos");
        double tolerance = 0.5;
        mjtNum current_goal_pos[3];
        mju_copy3(current_goal_pos, data->mocap_pos);

        mjtNum goal_displacement[3];
        mju_sub3(goal_displacement, current_goal_pos, current_pos);
        mjtNum goal_distance = mju_norm3(goal_displacement);
        if (goal_distance < tolerance)
        {
            current_goal_pos[0] += 3;
            mju_copy3(data->mocap_pos, current_goal_pos);
        }
    }
} // namespace mjpc
