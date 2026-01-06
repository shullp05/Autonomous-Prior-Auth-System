
import { describe, it, expect } from 'vitest';
import { computeMetrics, formatHours, formatCurrency } from '../metricsEngine';
import { RAW_STATUS } from '../statusConfig';

describe('metricsEngine Contract', () => {

    const mockData = [
        { status: RAW_STATUS.APPROVED, value: 5000 },
        { status: RAW_STATUS.APPROVED, value: 3000 },
        { status: RAW_STATUS.DENIED, value: 1000 },
        { status: RAW_STATUS.MANUAL_REVIEW, value: 2000 }, // Needs Clarification (via fallback)
        { status: RAW_STATUS.PROVIDER_ACTION_REQUIRED, value: 1500 }, // Missing Data
        { status: RAW_STATUS.CDI_REQUIRED, value: 8000 }, // CDI
    ];
    // Total Items: 6
    // Auto-Resolved: 2 Approved + 1 Denied = 3
    // Needs Review: 1 Manual + 1 Missing Data = 2
    // CDI: 1 (Separate bucket)

    const result = computeMetrics(mockData, { minutesPerCasePoint: 30 }); // 30 min per case = 0.5hr

    it('calculates auto_resolved count correctly', () => {
        // 2 Approved + 1 Denied. CDI excluded.
        expect(result.autoResolvedCount).toBe(3);
    });

    it('calculates needs_review_total as sum of subtypes', () => {
        // Manual + Provider Action
        expect(result.needsReviewTotal).toBe(2);
        expect(result.needsClarificationCount).toBe(1);
        expect(result.missingRequiredDataCount).toBe(1);
    });

    it('isolates CDI Required as separate risk bucket', () => {
        expect(result.cdiRequiredCount).toBe(1);
        expect(result.revenueAtRiskValue).toBe(8000);
        expect(result.needsReviewTotal).not.toContain(result.cdiRequiredCount); // Important contract!
    });

    it('computes hours saved linearly based on auto-resolved count', () => {
        // 3 auto-resolved * (30 min / 60) = 1.5 hours
        expect(result.hoursSavedPointRaw).toBe(1.5);
        expect(result.display.hoursSavedPoint).toBe('1.5');
    });

    it('calculates total value accurately', () => {
        // 5000 + 3000 + 1000 + 2000 + 1500 + 8000 = 20500
        expect(result.totalValue).toBe(20500);
    });

    it('formats sensitivity range string correctly', () => {
        // default min=15 (0.25hr), max=45 (0.75hr)
        // 3 cases * 0.25 = 0.75 hrs
        // 3 cases * 0.75 = 2.25 hrs
        expect(result.display.hoursSavedRange).toBe('0.8–2.3'); // Fixed to 1 decimal
    });

    it('handles empty population purely safely', () => {
        const empty = computeMetrics([]);
        expect(empty.totalScreened).toBe(0);
        expect(empty.display.hoursSavedPoint).toBe('0.0');
        expect(empty.display.revenueAtRisk).toBe('$0');
    });
});

describe('Formatters', () => {
    it('formats currency with suffixes', () => {
        expect(formatCurrency(1500000)).toBe('$1.5M');
        expect(formatCurrency(2500)).toBe('$2.5K');
        expect(formatCurrency(500)).toBe('$500');
    });

    it('formats hours with rounding', () => {
        expect(formatHours(1.5555)).toBe('1.6');
        expect(formatHours(0)).toBe('0.0');
    });
});
