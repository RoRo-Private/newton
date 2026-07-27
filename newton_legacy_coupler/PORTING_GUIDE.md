# Porting Guide: Genesis LegacyCoupler -> Newton Physics

## 1. Core mapping

Genesis primitive | Newton-side equivalent
- SDF signed distance/normal | closest-point + custom distance field or convex distance query
- rigid point velocity | body linear velocity + angular velocity cross lever arm
- apply coupling force | add force/torque to rigid body accumulator
- per-substep couple loop | world update callback before integration finalize

## 2. Recommended integration point

Inject coupler calls per substep:

1. local solver updates (MPM/SPH/PBD/FEM)
2. `coupler.Preprocess(substep)`
3. `coupler.Couple(substep)`
4. finalize rigid integration

This mirrors Genesis ordering and avoids one-frame lag.

## 3. Minimal MPM-rigid path first

Implement only:
- `IsRigidActive`, `IsMpmActive`, `SubstepDt`
- `QueryRigidCollision`
- `RigidVelocityAtPoint`
- `ApplyCouplingForce`
- `CoupleMpmWithRigid`

Inside `CoupleMpmWithRigid`:

1. iterate candidate MPM sample points (grid nodes or particles)
2. call `QueryRigidCollision(pos)`
3. compute new velocity via `ResolveRigidCollision(pos, vel, mass, sample)`
4. write velocity back

## 4. Stability controls (start conservative)

- Keep `coup_restitution` near `0.0`
- Keep `coup_friction` low to medium (`0.0 ~ 0.2`)
- Use moderate `coup_softness`; too small can make interaction stiff/noisy
- Clamp inverse dt for corrective velocity style paths if you add PBD attach mode

## 5. Validation checklist

- Momentum reaction applied to rigid side (not one-way)
- No NaN in velocity/force
- Contact remains stable at high mass ratio
- Deterministic behavior across repeated runs with same seed

## 6. Next extension order

1. MPM <-> rigid
2. SPH <-> rigid
3. PBD <-> rigid
4. non-rigid <-> non-rigid (MPM/SPH/PBD/FEM)

