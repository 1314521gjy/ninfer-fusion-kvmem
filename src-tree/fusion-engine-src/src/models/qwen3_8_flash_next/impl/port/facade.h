#pragma once

// What a ported Flash-Next load-layer translation unit needs in order to compile against this
// engine, in one place.
//
// THE ONE NAME THAT CANNOT BE ALIASED
//   The ported load layer says `artifact::Binder`, and so does this engine. Those are two different
//   classes -- the port's adapter (`ninfer::port::Binder`) and the engine's own `artifact::Binder`.
//   One name cannot denote both, so `Binder` is the single call-site edit a ported file needs:
//
//       artifact::Binder&          ->  ninfer::port::Binder&
//
//   The adapter keeps the engine binder reachable through `engine()`, so nothing is lost.
//
// WHAT DOES GET ALIASED
//   The names that exist only on the upstream side, or that upstream spells differently and this
//   engine has no conflicting meaning for:
//
//     NumericFormat   upstream's name for this engine's QType
//     StorageLayout   upstream's layout enum; this engine carries layouts as QuantLayout
//     TensorPlacement upstream's device/validate switch
//     ResourceEncoding, PayloadSpan
//     materialized_tensor / materialized_weight / artifact_object_offset
//
//   `ObjectHandle`, `Binding`, `Part`, `MaterializedArtifact`, `MaterializationPlan`,
//   `MaterializationStats`, `Reader` and `ArtifactError` are the SAME things in both generations, so
//   they keep resolving to `ninfer::artifact` untouched.
//
// HOW TO ACTIVATE
//   A ported file includes this header and, inside its own `detail` namespace, writes:
//
//       using ninfer::port::Binder;
//       using namespace ninfer::portalias;
//
//   Forgetting to is a loud failure ("no member named ..."), never a silent wrong pick.

// The alias block itself is declared once, in `artifact/typed_binding.h`, because the ported load
// layer and upstream's copied typed-binding unit must agree on it. Repeating the list here would be
// one fact with two homes, so this header only activates it.
#include "artifact/typed_binding.h"

using namespace ninfer::portalias;

