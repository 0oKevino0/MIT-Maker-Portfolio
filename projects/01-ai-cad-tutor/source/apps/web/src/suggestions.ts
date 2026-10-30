// Suggested questions for the empty state.
//
// Each one is answerable from material the internal demo is actually authorized to retrieve: the
// frozen experimental concept corpus (solidworks_concepts_v0 carries Chamfer, Fillet, Reference
// Plane and Shell, among its 24 terms) and the single Shell tutorial record admitted for INTERNAL
// DEMO RAG. They are wording the built-in help reply already suggests. None is a frozen evaluation
// question — checked against the evaluation sets on 2026-10-10 before this list was written — and
// none promises a capability the demo does not have (no SOLIDWORKS control, no drawing reading).
//
// A suggestion only fills the composer; the student still presses send.

export const SUGGESTED_QUESTIONS: readonly string[] = [
  "How do I use Shell?",
  "What is a Chamfer?",
  "How is a Fillet different from a Chamfer?",
  "What is a Reference Plane?",
];
