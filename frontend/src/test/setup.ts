import fc from "fast-check";

// Enforce the project-wide minimum of 100 iterations per property test
// (see tasks.md: "Minimum 100 iterations per property test").
fc.configureGlobal({ numRuns: 100 });
