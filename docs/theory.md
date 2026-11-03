# Theory of operation

## Forward model (emission-time correct)

For a stationary observer at ECEF position `x`, an aircraft trajectory `p(t_e)` with velocity
`v(t_e)` (both evaluated at the **emission** time), and an effective speed of sound `c`:

```
r(t_e)        = |p(t_e) - x|
u(t_e)        = (p(t_e) - x) / r
v_toward(t_e) = -v(t_e) · u(t_e)              (> 0 approaching)
t_receive     = t_e + r(t_e) / c
f_receive     = f_source(t_e) · c / (c - v_toward(t_e))
```

`dt_receive/dt_e = 1 - v_toward/c > 0` for subsonic flight, so `t_receive(t_e)` is monotonic and
can be inverted by interpolation (`physics/doppler.py`). Evaluating the geometry at reception
time and shifting by a constant delay is wrong for curved or accelerating tracks; the unit test
`test_emission_vs_reception_geometry_differs_on_curved_track` shows differences above 0.5 Hz
(two spectrogram bins) for a 3 km-radius turn.

Trajectories are piecewise-linear in ECEF between ADS-B samples with smoothed finite-difference
velocities; no extrapolation beyond the samples or across gaps > 120 s. The closest point of
approach (CPA) is found exactly on each linear piece, not snapped to a sample. The **delayed
CPA** is the reception time of the emission at minimum slant range; it is what the recording
shows as the steepest point of the Doppler ridge.

`c = 331.3 + 0.606·T(°C)`; 343.4 m/s at 20 °C. Wind, refraction and gradients are not modelled;
an uncertainty on `c` is carried (≈10 m/s when temperature is unknown).

## Independent simulator (no inverse crime)

`audio/synthetic.py` synthesises the received pressure from a *cubic-spline* path by solving the
retarded-time equation per sample with Newton's method and evaluating the harmonic phase at the
emission time. It shares no code with the forward model and never uses the Doppler formula; the
Doppler shift emerges from the retarded phase. Synthetic flights are exported as ADS-B-like
samples (irregular cadence, feet, barometric altitude deliberately offset, dropouts) and flow
through the real ingestion path.

## Event features and kinematic bounds

From a traced ridge (`search/events.py`): the steepest-descent time (≈ delayed CPA), the received
frequency there (≈ `h·f0`), the approach/recede plateaus with ratio `R = f_app/f_rec` and the
maximum slope. For a straight flyby `v_r = c(R-1)/(R+1)` (a *lower* bound on aircraft speed:
the radial component) and `d = f_CPA v² / (c |df/dt|)`. Bounds are made generous on purpose.

## Candidate generation (coarse)

Segments overlapping the window/bbox/speed bins → 200 m ground grid in a local ENU frame with
DEM elevation (observer HAE = orthometric ground + EGM96 undulation + 1.5 m) → per segment,
delayed CPA at every grid point, kept only within the slant-range bounds (sparse table) →

* **Mode A** (unknown time): pairs of entries at the same point with `tau_B - tau_A = Δt ± tol`,
  `Δt` being the measured separation of the two steepest points; same-flight pairs excluded.
* **Mode B** (known time): `|tau - (t0 + s_event)| ≤ sigma_t0 + tol`, one segment per event.

Every stage records candidate counts. When more Mode A pairs exist than
`quality_settings.max_assignments` (dense real traffic: 10⁷ CPA entries for 1,300 segments over
a 29k-point grid), the pairs kept are those with the best *slant-distance consistency prior*
(`|log(r_geom / d_est)|` per event with `d_est = f_c v² / (c |df/dt|)` from the flight's actual
CPA speed), never simply the first ones in grid order; a warning is attached to the run.

## Scoring (expensive stage) and the locked alignment

For a hypothesis (point, segment per event) the predicted curve `(t_receive, D)` with
`D = c/(c - v_toward)` is sampled along the whitened spectrogram with bilinear interpolation for
each harmonic `h` at `f = f0 (1 + α(t - t_c)) h D(t)`; the score is the harmonic-weighted mean
of the robustly clipped whitened value (0 = background, 1 = one robust σ). `f0` is swept in a
narrow band around the ridge-derived estimate `median(f_ridge / (h D))` (then polished to 0.04 %
because the highest harmonic needs it), `α` (RPM drift) is swept with a quadratic penalty.

Curve samples are **weighted by the local Doppler slope** (floor 0.3): plateaus constrain only
the radial speed and are shared by whole families of hypotheses, whereas the CPA transition and
any manoeuvre-induced wiggle constrain distance and timing. Without this weighting a straight
distractor with a similar speed/distance ratio fitted a long-plateau event almost as well as the
true turning flight (observed in the synthetic benchmark case `long_sep_1100`).

**One shared clock origin `t0`** is swept over a small alignment window and chosen from event 1
(SPEC default; `t0_fit="joint"` is available). Event 2 is scored at exactly that `t0`. Allowing
event 2 its own shift would let a geometrically inconsistent flight pair score as well as the
truth — see `tests/unit/test_locked_scoring.py`.

Three further terms were added after false matches on real traffic (see the implementation
report, M7):

* **excursion penalty** `6·|log(R_pred/R_obs)|` where `R_obs` is the observed plateau ratio
  `f_max/f_min` of the ridge and `R_pred = max D / min D` of the predicted curve over the event
  window — a velocity-plausibility check that stops far-away (nearly flat) predicted curves from
  "fitting" a ridge that visibly sweeps by tens of percent;
* **coverage penalty** `3·max(0, 0.4 − coverage)` per event;
* **imbalance**: `combined = mean − 0.5·(max − min)/2` so one perfectly fitted event cannot carry
  a failed one (the signature of every false match observed: 7.2 / 1.1).

Total = combined − soft silence penalty (other audible-range flights between the
events with no ridge evidence, weighted by audibility and aircraft class, capped) − drift penalty
+ small class prior (jets −0.4; unknown type −0.1, never excluded).

## Search funnel

With real traffic the number of Mode A pairs reaches 5·10⁵; the funnel is: ultra-coarse screen
of **every** pair (2 harmonics, 3 f0 values, 4 s shift step, ≤150 curve samples, ~1.5 ms per
pair) → top 40,000 → coarse (3 harmonics, 1.5 % f0 step, 2 s shifts) → top 400 → fine (full
sweep, drift, silence) → top 5 distinct basins → 100 m / 25 m refinement. Candidate counts for
every stage are reported with the result.

## Refinement, ranking, abstention

Top distinct basins (non-maximum suppression at 3× grid) are refined on 100 m then 25 m grids,
recomputing terrain, CPA and curves per point. Basins that converge are merged. The
`estimated_uncertainty_m` is the radius of the region within 0.15 z of the best — a description
of the score landscape, **not** a calibrated confidence region. Abstention reasons:
`no_flight_coverage`, `no_candidate_geometry`, `weak_fit`, `poor_coverage`. An `ambiguous`
flag is raised when a distinct runner-up is within 0.3 z.

## Known degeneracies

* One straight constant-speed flyby with known time: translating the observer along the track
  is equivalent to shifting the clock, so a clock uncertainty of σ seconds maps to ±σ·v metres
  along-track; the side of the track is a mirror ambiguity. Two events, a curved track, or a
  tight clock remove it.
* Mode A with two parallel straight tracks has a similar one-parameter family; the silence term
  and curve shapes (slant distance) break it only partially. Multiple distinct maxima are
  reported rather than hidden.

## Automatic ridge detection in dense traffic

The Viterbi tracker extracts the best path per iteration; with several simultaneous aircraft
that path tends to "walk" from one aircraft's plateau to another's through noise, because a
few small frequency steps cost less than an OFF/ON switch. Paths are therefore split wherever
they run through sub-threshold values for longer than 1 s, pieces are re-joined only across tiny
gaps with frequency *and* slope continuity, and harmonic groups are anchored on a base ridge
(members must overlap it in time and show a stable small-rational frequency ratio) instead of
being unioned transitively. The event-pair picker penalises ridges with more than one steep
descent (merged aircraft). On a semi-synthetic recording with five real aircraft this turned a
single 1,285 s "mega-event" into the two genuine flybys plus the plateau remnants of the others.
