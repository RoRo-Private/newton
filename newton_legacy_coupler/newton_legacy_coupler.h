#pragma once

#include <array>
#include <cstdint>

namespace newton_port {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vec3 operator+(const Vec3& r) const { return {x + r.x, y + r.y, z + r.z}; }
    Vec3 operator-(const Vec3& r) const { return {x - r.x, y - r.y, z - r.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
};

double Dot(const Vec3& a, const Vec3& b);
double Norm(const Vec3& v);
Vec3 Normalize(const Vec3& v, double eps = 1e-12);

struct CouplerOptions {
    bool rigid_mpm{true};
    bool rigid_sph{true};
    bool rigid_pbd{true};
    bool rigid_fem{true};
    bool mpm_sph{true};
    bool mpm_pbd{true};
    bool fem_mpm{true};
    bool fem_sph{true};

    double eps{1e-12};
};

struct RigidCouplingMaterial {
    bool needs_coup{true};
    double coup_friction{0.1};
    double coup_softness{0.002};
    double coup_restitution{0.0};
};

struct CollisionSample {
    bool valid{false};
    double signed_distance{0.0};
    Vec3 normal_world{};
    std::int32_t rigid_shape_id{-1};
    std::int32_t rigid_body_id{-1};
    RigidCouplingMaterial material{};
};

class INewtonCouplerAdapter {
public:
    virtual ~INewtonCouplerAdapter() = default;

    virtual bool IsRigidActive() const = 0;
    virtual bool IsMpmActive() const = 0;
    virtual bool IsSphActive() const = 0;
    virtual bool IsPbdActive() const = 0;
    virtual bool IsFemActive() const = 0;

    virtual double SubstepDt() const = 0;

    // Query closest rigid collision sample around world point.
    virtual CollisionSample QueryRigidCollision(const Vec3& pos_world) const = 0;

    // Rigid velocity at world point.
    virtual Vec3 RigidVelocityAtPoint(std::int32_t rigid_body_id, const Vec3& pos_world) const = 0;

    // Apply reaction force to rigid side.
    virtual void ApplyCouplingForce(
        std::int32_t rigid_body_id,
        const Vec3& force_world,
        const Vec3& at_pos_world) = 0;

    // Iterate and mutate your non-rigid side states.
    // Implement these as no-op if not used in your project.
    virtual void CoupleMpmWithRigid() = 0;
    virtual void CoupleSphWithRigid() = 0;
    virtual void CouplePbdWithRigid() = 0;
    virtual void CoupleFemWithRigid() = 0;
    virtual void CoupleMpmSph() = 0;
    virtual void CoupleMpmPbd() = 0;
    virtual void CoupleFemMpm() = 0;
    virtual void CoupleFemSph() = 0;
};

class LegacyLikeCoupler {
public:
    LegacyLikeCoupler(INewtonCouplerAdapter& adapter, CouplerOptions options);

    void Build();
    void Preprocess(std::int32_t substep_index);
    void Couple(std::int32_t substep_index);

    // Utility: resolve one collision response exactly in Legacy style intent.
    Vec3 ResolveRigidCollision(
        const Vec3& pos_world,
        const Vec3& vel,
        double mass,
        const CollisionSample& sample) const;

private:
    INewtonCouplerAdapter& adapter_;
    CouplerOptions options_;

    bool rigid_mpm_{false};
    bool rigid_sph_{false};
    bool rigid_pbd_{false};
    bool rigid_fem_{false};
    bool mpm_sph_{false};
    bool mpm_pbd_{false};
    bool fem_mpm_{false};
    bool fem_sph_{false};

    double InfluenceFromSignedDistance(double signed_distance, double coup_softness) const;
};

}  // namespace newton_port
