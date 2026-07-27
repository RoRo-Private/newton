#include "newton_legacy_coupler.h"

#include <algorithm>
#include <cmath>

namespace newton_port {

double Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double Norm(const Vec3& v) {
    return std::sqrt(Dot(v, v));
}

Vec3 Normalize(const Vec3& v, double eps) {
    const double n = Norm(v);
    if (n < eps) {
        return {0.0, 0.0, 0.0};
    }
    return v / n;
}

LegacyLikeCoupler::LegacyLikeCoupler(INewtonCouplerAdapter& adapter, CouplerOptions options)
    : adapter_(adapter), options_(options) {}

void LegacyLikeCoupler::Build() {
    rigid_mpm_ = adapter_.IsRigidActive() && adapter_.IsMpmActive() && options_.rigid_mpm;
    rigid_sph_ = adapter_.IsRigidActive() && adapter_.IsSphActive() && options_.rigid_sph;
    rigid_pbd_ = adapter_.IsRigidActive() && adapter_.IsPbdActive() && options_.rigid_pbd;
    rigid_fem_ = adapter_.IsRigidActive() && adapter_.IsFemActive() && options_.rigid_fem;
    mpm_sph_ = adapter_.IsMpmActive() && adapter_.IsSphActive() && options_.mpm_sph;
    mpm_pbd_ = adapter_.IsMpmActive() && adapter_.IsPbdActive() && options_.mpm_pbd;
    fem_mpm_ = adapter_.IsFemActive() && adapter_.IsMpmActive() && options_.fem_mpm;
    fem_sph_ = adapter_.IsFemActive() && adapter_.IsSphActive() && options_.fem_sph;
}

void LegacyLikeCoupler::Preprocess(std::int32_t /*substep_index*/) {
    // Hook for CPIC-like normal cache or neighbor cache refresh.
    // Keep empty unless your non-rigid backend needs a pre-coupling pass.
}

void LegacyLikeCoupler::Couple(std::int32_t /*substep_index*/) {
    // Keep order close to Genesis LegacyCoupler intent.
    if (rigid_mpm_) {
        adapter_.CoupleMpmWithRigid();
    }
    if (rigid_sph_) {
        adapter_.CoupleSphWithRigid();
    }
    if (rigid_pbd_) {
        adapter_.CouplePbdWithRigid();
    }
    if (rigid_fem_) {
        adapter_.CoupleFemWithRigid();
    }
    if (mpm_sph_) {
        adapter_.CoupleMpmSph();
    }
    if (mpm_pbd_) {
        adapter_.CoupleMpmPbd();
    }
    if (fem_mpm_) {
        adapter_.CoupleFemMpm();
    }
    if (fem_sph_) {
        adapter_.CoupleFemSph();
    }
}

double LegacyLikeCoupler::InfluenceFromSignedDistance(double signed_distance, double coup_softness) const {
    const double softness = std::max(coup_softness, 1e-10);
    return std::min(std::exp(-signed_distance / softness), 1.0);
}

Vec3 LegacyLikeCoupler::ResolveRigidCollision(
    const Vec3& pos_world,
    const Vec3& vel,
    double mass,
    const CollisionSample& sample) const {
    if (!sample.valid || !sample.material.needs_coup) {
        return vel;
    }

    const double influence = InfluenceFromSignedDistance(sample.signed_distance, sample.material.coup_softness);
    if (influence <= 0.1) {
        return vel;
    }

    const Vec3 normal = Normalize(sample.normal_world, options_.eps);
    const Vec3 vel_rigid = adapter_.RigidVelocityAtPoint(sample.rigid_body_id, pos_world);

    // Relative velocity particle w.r.t rigid.
    const Vec3 rvel = vel - vel_rigid;
    const double rvel_n_mag = Dot(rvel, normal);

    // Only resolve inward collision.
    if (rvel_n_mag >= 0.0) {
        return vel;
    }

    // Tangential component with Coulomb-like clamping.
    Vec3 rvel_tan = rvel - normal * rvel_n_mag;
    const double tan_norm = std::max(Norm(rvel_tan), options_.eps);
    const double tan_after = std::max(0.0, tan_norm + rvel_n_mag * sample.material.coup_friction);
    rvel_tan = rvel_tan * (tan_after / tan_norm);

    // Normal restitution.
    const Vec3 rvel_normal = normal * (-rvel_n_mag * sample.material.coup_restitution);

    const Vec3 rvel_new = rvel_tan + rvel_normal;
    const Vec3 new_vel = vel_rigid + rvel_new * influence + rvel * (1.0 - influence);

    // Reaction force back to rigid side.
    const Vec3 delta_mv = (new_vel - vel) * mass;
    const double inv_dt = 1.0 / std::max(adapter_.SubstepDt(), options_.eps);
    const Vec3 reaction_force = delta_mv * (-inv_dt);
    adapter_.ApplyCouplingForce(sample.rigid_body_id, reaction_force, pos_world);

    return new_vel;
}

}  // namespace newton_port
