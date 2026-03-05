// cloakroom_baseline.pm
// Continuous-Time Markov Chain (CTMC) model of cloakroom swarm.
// Parameters derived from baseline LF simulation data.
// Probability levels L1–L5 via Equal Width Discretisation (EWD).

ctmc

// ── Constants: probability levels ───────────────────────────────────────────
const double L1 = 0.1000;
const double L2 = 0.3000;
const double L3 = 0.5000;
const double L4 = 0.7000;
const double L5 = 0.9000;

// ── State indices ────────────────────────────────────────────────────────────
// 0=SEARCHING  1=PICKUP  2=DROPOFF
// 3=AVOIDANCE_S  4=AVOIDANCE_P  5=AVOIDANCE_D

// ── Transition rates (derived from EWD levels) ───────────────────────────────
const double r_search_to_pickup   = 0.1000;    // L1
const double r_pickup_to_dropoff  = 0.3000;    // L2
const double r_dropoff_to_search  = 0.1000;    // L1
const double r_to_avoid           = 0.1000;  // L1
const double r_from_avoid         = 0.5;         // fixed return rate

// ── Module: swarm (counting abstraction, N robots) ───────────────────────────
// n_s = # robots in SEARCHING, etc.

module swarm

  n_s : [0..5] init 5;
  n_p : [0..5] init 0;
  n_d : [0..5] init 0;
  n_as: [0..5] init 0;
  n_ap: [0..5] init 0;
  n_ad: [0..5] init 0;

  // SEARCHING → PICKUP  (one robot per transition)
  [] n_s > 0 -> n_s * r_search_to_pickup :
      (n_s' = n_s - 1) & (n_p' = n_p + 1);

  // PICKUP → DROPOFF
  [] n_p > 0 -> n_p * r_pickup_to_dropoff :
      (n_p' = n_p - 1) & (n_d' = n_d + 1);

  // DROPOFF → SEARCHING (carrier deposited)
  [] n_d > 0 -> n_d * r_dropoff_to_search :
      (n_d' = n_d - 1) & (n_s' = n_s + 1);

  // SEARCHING → AVOIDANCE_S
  [] n_s > 0 -> n_s * r_to_avoid :
      (n_s' = n_s - 1) & (n_as' = n_as + 1);

  // PICKUP → AVOIDANCE_P
  [] n_p > 0 -> n_p * r_to_avoid :
      (n_p' = n_p - 1) & (n_ap' = n_ap + 1);

  // DROPOFF → AVOIDANCE_D
  [] n_d > 0 -> n_d * r_to_avoid :
      (n_d' = n_d - 1) & (n_ad' = n_ad + 1);

  // AVOIDANCE_S → SEARCHING
  [] n_as > 0 -> n_as * r_from_avoid :
      (n_as' = n_as - 1) & (n_s' = n_s + 1);

  // AVOIDANCE_P → PICKUP
  [] n_ap > 0 -> n_ap * r_from_avoid :
      (n_ap' = n_ap - 1) & (n_p' = n_p + 1);

  // AVOIDANCE_D → DROPOFF
  [] n_ad > 0 -> n_ad * r_from_avoid :
      (n_ad' = n_ad - 1) & (n_d' = n_d + 1);

endmodule

// ── Safety labels ─────────────────────────────────────────────────────────────
// Probabilities derived from simulation data.

// Approximation: probability of any robot entering red zone at given state
// is modelled as a separate independent process with rate r_unsafe_red.
const double r_unsafe_red            = 0.140000;   // 14.0%
const double r_unsafe_amber_critical = 0.084000; // 8.4%
const double r_req2_viol             = 0.000000;          // 0.0%

module safety_monitor

  unsafe_red   : bool init false;
  unsafe_amber : bool init false;
  unsafe_req2  : bool init false;

  [] !unsafe_red   -> r_unsafe_red            : (unsafe_red'   = true);
  [] !unsafe_amber -> r_unsafe_amber_critical : (unsafe_amber' = true);
  [] !unsafe_req2  -> r_req2_viol             : (unsafe_req2'  = true);

endmodule

// ── Rewards ───────────────────────────────────────────────────────────────────
rewards "main_states"
  n_s > 0 | n_p > 0 | n_d > 0 : n_s + n_p + n_d;
endrewards

rewards "avoidance_states"
  n_as > 0 | n_ap > 0 | n_ad > 0 : n_as + n_ap + n_ad;
endrewards

// ── Labels ────────────────────────────────────────────────────────────────────
label "unsafe_fireexitsblocked" = unsafe_red = true;
label "unsafe_amber_critical"   = unsafe_amber = true;
label "unsafe_amber"            = unsafe_amber = true;
label "unsafe_red"              = unsafe_red = true;
label "unsafe_density"          = unsafe_req2 = true;
