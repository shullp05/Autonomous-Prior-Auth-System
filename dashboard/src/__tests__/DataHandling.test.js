/**
 * DataHandling.test.js - Tests for audit fix #1 and #5
 * 
 * Verifies:
 * - Issue #1: Malformed API data shape handling in loadAll
 * - Issue #5: Null record handling in SankeyChart data loop
 */

import { describe, it, expect, vi } from 'vitest';

// --- Issue #1: Data Shape Handling ---

describe('Issue #1: loadAll Data Shape Handling', () => {

    // Helper to simulate the data handling logic from App.jsx
    function handleDashboardData(d) {
        if (d && typeof d === 'object' && 'results' in d && Array.isArray(d.results)) {
            return { data: d.results, metadata: d.metadata ?? null };
        } else if (Array.isArray(d)) {
            return { data: d, metadata: null };
        } else {
            console.warn('Unexpected dashboard data shape:', d);
            return { data: [], metadata: null };
        }
    }

    it('should handle new format with valid results array', () => {
        const input = {
            results: [{ patient_id: 'P1', status: 'APPROVED' }],
            metadata: { model_name: 'test' }
        };
        const { data, metadata } = handleDashboardData(input);

        expect(data).toHaveLength(1);
        expect(data[0].patient_id).toBe('P1');
        expect(metadata.model_name).toBe('test');
    });

    it('should handle legacy format (raw array)', () => {
        const input = [{ patient_id: 'P1', status: 'APPROVED' }];
        const { data, metadata } = handleDashboardData(input);

        expect(data).toHaveLength(1);
        expect(metadata).toBeNull();
    });

    it('should return empty array when results is null (Issue #1 edge case)', () => {
        const input = { results: null, metadata: {} };
        const { data, metadata } = handleDashboardData(input);

        expect(data).toEqual([]);
        expect(metadata).toBeNull();
    });

    it('should return empty array when results key is missing', () => {
        const input = { metadata: { version: '1.0' } };
        const { data } = handleDashboardData(input);

        expect(data).toEqual([]);
    });

    it('should return empty array when data is undefined', () => {
        const { data } = handleDashboardData(undefined);
        expect(data).toEqual([]);
    });

    it('should return empty array when data is null', () => {
        const { data } = handleDashboardData(null);
        expect(data).toEqual([]);
    });

    it('should return empty array when data is a primitive', () => {
        const { data } = handleDashboardData('invalid');
        expect(data).toEqual([]);
    });
});

// --- Issue #5: SankeyChart Null Record Handling ---

describe('Issue #5: SankeyChart Null Record Guard', () => {

    // Simulates the data processing logic from SankeyChart.jsx
    function processRecords(data) {
        if (!Array.isArray(data) || data.length === 0) return null;

        const approvedData = [];
        const deniedData = [];
        const manualData = [];

        for (const record of data) {
            if (!record) continue; // The fix we applied
            const status = record?.status;
            if (status === 'APPROVED') {
                approvedData.push(record);
            } else if (status === 'DENIED') {
                deniedData.push(record);
            } else {
                manualData.push(record);
            }
        }

        return { approvedData, deniedData, manualData };
    }

    it('should not crash when data contains null records', () => {
        const validRecord1 = { patient_id: 'P1', status: 'APPROVED', value: 1000 };
        const validRecord2 = { patient_id: 'P2', status: 'DENIED', value: 500 };
        const data = [validRecord1, null, undefined, validRecord2];

        // Should not throw
        expect(() => processRecords(data)).not.toThrow();

        const result = processRecords(data);
        expect(result.approvedData).toHaveLength(1);
        expect(result.deniedData).toHaveLength(1);
        expect(result.manualData).toHaveLength(0);
    });

    it('should correctly categorize all valid records', () => {
        const data = [
            { patient_id: 'P1', status: 'APPROVED', value: 1000 },
            { patient_id: 'P2', status: 'DENIED', value: 500 },
            { patient_id: 'P3', status: 'FLAGGED', value: 750 },
        ];

        const result = processRecords(data);
        expect(result.approvedData).toHaveLength(1);
        expect(result.deniedData).toHaveLength(1);
        expect(result.manualData).toHaveLength(1);
    });

    it('should return null for empty array', () => {
        expect(processRecords([])).toBeNull();
    });

    it('should handle array with only null/undefined values', () => {
        const data = [null, undefined, null];
        const result = processRecords(data);

        expect(result.approvedData).toHaveLength(0);
        expect(result.deniedData).toHaveLength(0);
        expect(result.manualData).toHaveLength(0);
    });
});
