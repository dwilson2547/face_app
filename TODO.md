# TODO

Backlog imported from the retired todo store, 2026-07-06.

## Medium

- [ ] **Review face app UI in detail** — Deeper walkthrough of the new review workstation UI, including correction and recluster flows, and capture any UX or workflow gaps.
- [ ] **Analyze next face app steps** — Assess the best next implementation steps after the review workstation milestone, including workflow refinements, architecture direction, and the next highest-value features.

## Unprioritized (imported from workman drafts, 2026-07-06)

- [ ] **Large photo library performance and scalability** — Profile and optimize for 10k+ photo libraries. Bottlenecks: sequential image loading, single-process detection, memory. Consider batching, multiprocessing, incremental run support.
- [ ] **EXIF metadata integration** — Extract and store EXIF (capture date, GPS, camera model) alongside face records. Enables date-range filtering. Errors must soft-fail.
- [ ] **Person export and photo album grouping** — Export workflow: group photos by person label into output/<label>/ directories. Unlabeled clusters go to output/unknown-<id>/.
