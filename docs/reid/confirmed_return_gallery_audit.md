# Confirmed return gallery provenance audit

`audit_confirmed_return_gallery.py --recovery-report <report.json>` audits every
confirmed event in a completed `confirmed_clip_return_paired_v1` run.

Phase 1 replays the enabled registry on frozen local tracks, segment cuts, raw
CLIP features and historical means. It reads no GT and executes no model or video
decoder. Every output record, archive decision, lifecycle counter and return must
exactly reproduce the source. Observer wrappers call the original functions once
and preserve their inputs and results; original modules are not edited.

At retirement, the observer captures the exact samples returned by the existing
gallery builder. At confirmed recovery it verifies that those samples reconstruct
the actual stored reference byte-for-byte, and that the current raw query vectors
reconstruct the runtime query. It reproduces the accepted cosine and records the
nearest geometry-feasible alternative, including below-threshold rivals. A null
margin means that no alternative was geometry-feasible, not infinite certainty.

Phase 2 runs only after selected samples are frozen. It joins effective segment
keys back to original observation keys, verifies each selected vector against its
persisted raw embedding row, and labels the original tracked boxes with GT. All
camera predictions participate in spatial uniqueness. ID zero is valid. Samples
outside the evaluation interval are explicitly unknown, never treated as empty
annotated frames. No sample is removed or reranked using GT.

The report records each selected sample's frame, camera, segment, original local
ID, confidence, normalized border clearance, ground projection, observed global
ID, cosine to the query and spatial evidence. Event summaries include reference
ages, the actual geometry distance limit, and gallery/query GT counts.

Interpretation:
- A mixed gallery can contain evidence from a local switch or a previous global
  merge. Appearance confidence is not identity certainty.
- An unknown GT label is neither evidence of the same person nor proof of error.
- High similarity between clean, differently labeled references and queries
  suggests an appearance confusion; visual verification may still be needed.
- This audit does not estimate the counterfactual IDF1 impact of removing an event.
- Native ground units are not assumed to be meters.

Files are written to `artifacts/confirmed_return_gallery/<timestamp>/`:
`unlabeled_samples.json` and `report.json`. Runtime parameters and predictions are
unchanged. Existing code must still match the hashes recorded by the source run.

Local verification: observer-owned samples and unchanged reference output;
serialized enabled replay from a 150-round paired-registry fixture, exact
reference/query reconstruction, persisted vector-row equality and GT-zero labels.
