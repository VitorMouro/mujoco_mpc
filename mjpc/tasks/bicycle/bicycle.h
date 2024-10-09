#ifndef MJPC_TASKS_BICYCLE_BICYCLE_H_
#define MJPC_TASKS_BICYCLE_BICYCLE_H_

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc
{
  class Bicycle : public Task
  {
  public:
    std::string Name() const override;
    std::string XmlPath() const override;

    class ResidualFn : public BaseResidualFn
    {
      friend class Bicycle;

    public:
      explicit ResidualFn(const Bicycle *task) : BaseResidualFn(task) {}

      void Residual(const mjModel *model, const mjData *data,
                    double *residual) const override;
    };

    Bicycle() : residual_(this) {}
    void TransitionLocked(mjModel *model, mjData *data) override;
    void ModifyScene(const mjModel *model, const mjData *data,
                     mjvScene *scene) const override;

  protected:
    std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override
    {
      return std::make_unique<ResidualFn>(this);
    }
    ResidualFn *InternalResidual() override { return &residual_; }

  private:
    ResidualFn residual_;
    bool locked = true;

  };
} // namespace mjpc

#endif // MJPC_TASKS_BICYCLE_BICYCLE_H_
