/**
 * metricsEngine.test.js - Unit tests for metrics contract module
 * 
 * Verifies:
 * - Correct metric calculations
 * - Status grouping consistency
 * - Formatting rules
 * - Sensitivity range computation
 */

import { describe, it, expect, test } from 'vitest';
import {
    computeMetrics,
    formatHours,
    formatCurrency,
    formatPercent,
    DEFAULT_CONFIG,
} from '../metricsEngine';

// =============================================================================
// Test Fixtures
// =============================================================================

const createRecord = (status, value = 1000, reason = '') => ({
    patient_id: `P-${Math.random().toString(36).slice(2, 8)}`,
    status,
    value,
    reason,
});

const SAMPLE_POPULATION = [
    createRecord('APPROVED', 1500),
    createRecord('APPROVED', 2000),
    createRecord('DENIED', 800, 'Does not meet BMI criteria'),
    createRecord('DENIED', 1200, 'HARD STOP: Safety exclusion'),
    createRecord('FLAGGED', 900),
    createRecord('PROVIDER_ACTION_REQUIRED', 1100),
    createRecord('MANUAL_REVIEW', 600),
];

// =============================================================================
// computeMetrics Tests
// =============================================================================

describe('computeMetrics', () => {

    describe('Count calculations', () => {

        test('totalScreened equals population length (excluding nulls)', () => {
            const population = [createRecord('APPROVED'), null, createRecord('DENIED')];
            const metrics = computeMetrics(population);
            expect(metrics.totalScreened).toBe(2);
        });

        test('approvedCount only includes APPROVED status', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.approvedCount).toBe(2);
        });

        test('deniedCount includes all DENIED variants', () => {
            const population = [
                createRecord('DENIED'),
                createRecord('DENIED_SAFETY'),
                createRecord('DENIED_CLINICAL'),
                createRecord('DENIED_MISSING_INFO'),
            ];
            const metrics = computeMetrics(population);
            expect(metrics.deniedCount).toBe(4);
        });

        test('needsClarificationCount includes FLAGGED and MANUAL_REVIEW', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.needsClarificationCount).toBe(2); // FLAGGED + MANUAL_REVIEW
        });

        test('missingRequiredDataCount includes PROVIDER_ACTION_REQUIRED', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.missingRequiredDataCount).toBe(1);
        });

    });

    describe('Derived metrics', () => {

        test('needsReviewTotal equals needsClarification + missingRequiredData', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.needsReviewTotal).toBe(
                metrics.needsClarificationCount + metrics.missingRequiredDataCount
            );
        });

        test('autoResolvedCount equals approved + denied', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.autoResolvedCount).toBe(
                metrics.approvedCount + metrics.deniedCount
            );
        });

        test('autoResolvedCount + needsReviewTotal equals totalScreened', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.autoResolvedCount + metrics.needsReviewTotal).toBe(
                metrics.totalScreened
            );
        });

        test('autoResolutionRate calculated correctly', () => {
            const population = [
                createRecord('APPROVED'),
                createRecord('APPROVED'),
                createRecord('DENIED'),
                createRecord('FLAGGED'),
            ];
            const metrics = computeMetrics(population);
            // 3 auto-resolved / 4 total = 0.75
            expect(metrics.autoResolutionRate).toBe(0.75);
        });

    });

    describe('Hours saved calculations', () => {

        test('hoursSavedPointRaw uses default 25 min/case', () => {
            const population = [
                createRecord('APPROVED'),
                createRecord('DENIED'),
            ];
            const metrics = computeMetrics(population);
            // 2 auto-resolved × 25/60 hours = 2 × 0.4167 ≈ 0.833
            const expected = 2 * (25 / 60);
            expect(metrics.hoursSavedPointRaw).toBeCloseTo(expected, 5);
        });

        test('hoursSavedMin uses 15 min/case', () => {
            const population = [createRecord('APPROVED'), createRecord('DENIED')];
            const metrics = computeMetrics(population);
            const expected = 2 * (15 / 60);
            expect(metrics.hoursSavedMin).toBeCloseTo(expected, 5);
        });

        test('hoursSavedMax uses 45 min/case', () => {
            const population = [createRecord('APPROVED'), createRecord('DENIED')];
            const metrics = computeMetrics(population);
            const expected = 2 * (45 / 60);
            expect(metrics.hoursSavedMax).toBeCloseTo(expected, 5);
        });

        test('custom config overrides default minutes', () => {
            const population = [createRecord('APPROVED')];
            const metrics = computeMetrics(population, { minutesPerCasePoint: 30 });
            const expected = 1 * (30 / 60);
            expect(metrics.hoursSavedPointRaw).toBeCloseTo(expected, 5);
        });

    });

    describe('Value calculations', () => {

        test('totalValue sums all record values', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            const expected = 1500 + 2000 + 800 + 1200 + 900 + 1100 + 600;
            expect(metrics.totalValue).toBe(expected);
        });

        test('approvedValue sums only approved records', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.approvedValue).toBe(1500 + 2000);
        });

        test('deniedValue sums only denied records', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.deniedValue).toBe(800 + 1200);
        });

    });

    describe('Safety contraindication detection', () => {

        test('safetyContraindicationCount detects HARD STOP in reason', () => {
            const population = [
                createRecord('DENIED', 1000, 'HARD STOP: Safety exclusion'),
                createRecord('DENIED', 1000, 'Does not meet BMI criteria'),
            ];
            const metrics = computeMetrics(population);
            expect(metrics.safetyContraindicationCount).toBe(1);
            expect(metrics.notEligibleCount).toBe(1);
        });

    });

    describe('Display values', () => {

        test('display.hoursSavedPoint is always 1 decimal', () => {
            const population = Array(7).fill(null).map(() => createRecord('APPROVED'));
            const metrics = computeMetrics(population);
            // 7 × 0.4167 = 2.9167 → should display as "2.9"
            expect(metrics.display.hoursSavedPoint).toMatch(/^\d+\.\d$/);
        });

        test('display.hoursSavedRange shows min–max format', () => {
            const population = [createRecord('APPROVED')];
            const metrics = computeMetrics(population);
            expect(metrics.display.hoursSavedRange).toMatch(/^\d+\.\d–\d+\.\d$/);
        });

        test('display.autoResolutionRate ends with %', () => {
            const metrics = computeMetrics(SAMPLE_POPULATION);
            expect(metrics.display.autoResolutionRate).toMatch(/%$/);
        });

    });

    describe('Edge cases', () => {

        test('empty population returns zeros', () => {
            const metrics = computeMetrics([]);
            expect(metrics.totalScreened).toBe(0);
            expect(metrics.autoResolvedCount).toBe(0);
            expect(metrics.hoursSavedPointRaw).toBe(0);
        });

        test('population with only nulls returns zeros', () => {
            const metrics = computeMetrics([null, null, undefined]);
            expect(metrics.totalScreened).toBe(0);
        });

        test('missing value field treated as 0', () => {
            const population = [{ status: 'APPROVED' }];
            const metrics = computeMetrics(population);
            expect(metrics.totalValue).toBe(0);
            expect(metrics.approvedCount).toBe(1);
        });

    });

});

// =============================================================================
// Formatting Helper Tests
// =============================================================================

describe('formatHours', () => {

    test('formats to 1 decimal by default', () => {
        expect(formatHours(2.567)).toBe('2.6');
        expect(formatHours(10.01)).toBe('10.0');
    });

    test('respects custom decimals parameter', () => {
        expect(formatHours(2.567, 2)).toBe('2.57');
        expect(formatHours(2.567, 0)).toBe('3');
    });

    test('returns em dash for null/undefined', () => {
        expect(formatHours(null)).toBe('—');
        expect(formatHours(undefined)).toBe('—');
    });

    test('returns em dash for non-finite numbers', () => {
        expect(formatHours(NaN)).toBe('—');
        expect(formatHours(Infinity)).toBe('—');
    });

});

describe('formatCurrency', () => {

    test('formats millions with M suffix', () => {
        expect(formatCurrency(1_500_000)).toBe('$1.5M');
        expect(formatCurrency(2_000_000)).toBe('$2.0M');
    });

    test('formats thousands with K suffix', () => {
        expect(formatCurrency(1_500)).toBe('$1.5K');
        expect(formatCurrency(250_000)).toBe('$250.0K');
    });

    test('formats small amounts as currency', () => {
        expect(formatCurrency(500)).toBe('$500');
        expect(formatCurrency(99)).toBe('$99');
    });

    test('returns em dash for null/undefined', () => {
        expect(formatCurrency(null)).toBe('—');
        expect(formatCurrency(undefined)).toBe('—');
    });

});

describe('formatPercent', () => {

    test('formats decimal as percentage', () => {
        expect(formatPercent(0.75)).toBe('75.0%');
        expect(formatPercent(0.333)).toBe('33.3%');
    });

    test('handles 0 and 1', () => {
        expect(formatPercent(0)).toBe('0.0%');
        expect(formatPercent(1)).toBe('100.0%');
    });

    test('returns em dash for null/undefined', () => {
        expect(formatPercent(null)).toBe('—');
        expect(formatPercent(undefined)).toBe('—');
    });

});
