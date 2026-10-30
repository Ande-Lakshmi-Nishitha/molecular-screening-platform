import { describe, expect, it } from "vitest";
import { canCancel, isActive, stateLabel, tierLabel } from "./jobState";

describe("jobState", () => {
  it("treats queued and running as active", () => {
    expect(isActive("queued")).toBe(true);
    expect(isActive("running")).toBe(true);
  });

  it("treats every final status as not active", () => {
    for (const s of ["partial", "complete", "failed", "cancelled"]) {
      expect(isActive(s)).toBe(false);
    }
  });

  it("offers cancel only while the job can still change (FR-056)", () => {
    expect(canCancel({ status: "queued" })).toBe(true);
    expect(canCancel({ status: "running" })).toBe(true);
    expect(canCancel({ status: "complete" })).toBe(false);
    expect(canCancel({ status: "partial" })).toBe(false);
    expect(canCancel({ status: "cancelled" })).toBe(false);
    expect(canCancel(null)).toBe(false);
  });

  it("labels tiers and states, falling back to the raw value", () => {
    expect(tierLabel("qml")).toBe("Quantum (QML)");
    expect(tierLabel("other")).toBe("other");
    expect(stateLabel("skipped")).toBe("Skipped");
    expect(stateLabel("complete")).toBe("Complete");
    expect(stateLabel("partial")).toBe("Partly complete");
    expect(stateLabel("weird")).toBe("weird");
  });
});
