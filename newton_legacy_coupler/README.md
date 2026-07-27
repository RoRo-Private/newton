# Newton Legacy Coupler Porting Kit

This folder is a **drop-in scaffold** to port Genesis `LegacyCoupler` ideas into a Newton Physics based project.

## Goals

- Mirror Genesis coupling flow: `build -> preprocess -> couple`
- Keep engine-specific code isolated behind a small adapter interface
- Support pair toggles similar to `LegacyCouplerOptions`

## Files

- `newton_legacy_coupler.h`: public API and data contracts
- `newton_legacy_coupler.cpp`: reference implementation of coupling loop
- `newton_adapter_example.h`: adapter hooks to connect your Newton repo APIs
- `PORTING_GUIDE.md`: step-by-step integration guide

## What you must wire in Newton repo

1. Signed distance + normal query for rigid shapes
2. Rigid point velocity at world point
3. Apply coupling force/torque back to rigid body
4. Particle/grid iteration hooks for your non-rigid side (SPH/MPM/PBD/FEM or custom)
5. Substep timing (`dt`) and world step loop integration

## Minimal integration order

1. Implement `INewtonCouplerAdapter`
2. Call `LegacyLikeCoupler::Build()` once after scene construction
3. Call `Preprocess(substep_idx)` each substep (optional path)
4. Call `Couple(substep_idx)` each substep after solver local updates

