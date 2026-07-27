#pragma once

#include "newton_legacy_coupler.h"

namespace newton_port {

// Replace all TODOs with real Newton API calls in your repo.
class NewtonAdapterExample final : public INewtonCouplerAdapter {
public:
    bool IsRigidActive() const override { return true; }
    bool IsMpmActive() const override { return true; }
    bool IsSphActive() const override { return false; }
    bool IsPbdActive() const override { return false; }
    bool IsFemActive() const override { return false; }

    double SubstepDt() const override {
        // TODO: return current Newton/world substep dt
        return 1.0 / 120.0;
    }

    CollisionSample QueryRigidCollision(const Vec3& /*pos_world*/) const override {
        // TODO: perform SDF or closest-point query from your collision world
        return {};
    }

    Vec3 RigidVelocityAtPoint(std::int32_t /*rigid_body_id*/, const Vec3& /*pos_world*/) const override {
        // TODO: body linear velocity + omega x r
        return {};
    }

    void ApplyCouplingForce(
        std::int32_t /*rigid_body_id*/,
        const Vec3& /*force_world*/,
        const Vec3& /*at_pos_world*/) override {
        // TODO: accumulate/apply force and torque to Newton body
    }

    void CoupleMpmWithRigid() override {
        // TODO: iterate MPM nodes/particles and call ResolveRigidCollision
    }
    void CoupleSphWithRigid() override {}
    void CouplePbdWithRigid() override {}
    void CoupleFemWithRigid() override {}
    void CoupleMpmSph() override {}
    void CoupleMpmPbd() override {}
    void CoupleFemMpm() override {}
    void CoupleFemSph() override {}
};

}  // namespace newton_port
