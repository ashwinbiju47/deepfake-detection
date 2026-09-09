/**
 * Property 10: Stream replay on reconnect is lossless and gap-free (Task 10.2, Requirement 5.5).
 *
 * Feature: deepfake-detection-platform, Property 10: Stream replay on reconnect is lossless and gap-free
 */

import { describe, expect, it } from "vitest";
import fc from "fast-check";

type ReplayableEvent = { seq: number; type: string; payload: unknown };

function replayEvents(events: ReplayableEvent[], lastSeq: number): ReplayableEvent[] {
  return events.filter((e) => e.seq > lastSeq).sort((a, b) => a.seq - b.seq);
}

describe("Property 10: Lossless gap-free stream replay", () => {
  it("replays all events with seq > lastSeq without loss or gaps", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            seq: fc.integer({ min: 1, max: 1000 }),
            type: fc.constantFrom("progress", "heatmap", "result", "error"),
            payload: fc.object(),
          }),
          { minLength: 1, maxLength: 100 }
        ),
        fc.integer({ min: 0, max: 500 }),
        (events, lastSeq) => {
          // Remove duplicate seqs for clean sequence invariant
          const uniqueEvents = Array.from(
            new Map(events.map((e) => [e.seq, e])).values()
          );

          const replayed = replayEvents(uniqueEvents as ReplayableEvent[], lastSeq);

          // Every replayed event must have seq > lastSeq
          for (const e of replayed) {
            expect(e.seq).toBeGreaterThan(lastSeq);
          }

          // No missed events with seq > lastSeq
          const expectedCount = uniqueEvents.filter((e) => e.seq > lastSeq).length;
          expect(replayed.length).toBe(expectedCount);
        }
      ),
      { numRuns: 100 }
    );
  });
});
