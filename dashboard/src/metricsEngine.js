/**
 * metricsEngine.js - Central metrics computation module
 * 
 * This module computes all dashboard metrics from a single source,
 * ensuring consistency between KPIs, tooltips, and displays.
 */

import {
    safeUpper,
    RAW_STATUS,
    REVIEW_BUCKET,
    getReviewBucket,
    isSafetyContraindication,
} from './statusConfig';

// =============================================================================
// Configuration
// =============================================================================

export const DEFAULT_CONFIG = {
    minutesPerCasePoint: 25,    // Default assumption (MGMA/AMA complex PA)
    minutesPerCaseMin: 15,      // Sensitivity range minimum
    minutesPerCaseMax: 45,      // Sensitivity range maximum
    hoursDecimals: 1,           // Decimal places for hours display
    currencyDecimals: 1,        // Decimal places for currency (K/M)
};

// =============================================================================
// Formatting Helpers (centralized to prevent drift)
// =============================================================================

/**
 * Format hours with consistent decimal places
 * @param {number} value - Hours value
 * @param {number} decimals - Decimal places (default from config)
 * @returns {string} Formatted string
 */
export function formatHours(value, decimals = DEFAULT_CONFIG.hoursDecimals) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return '—';
    }
    return Number(value).toFixed(decimals);
}

/**
 * Format currency with K/M suffix
 * @param {number} value - Dollar amount
 * @returns {string} Formatted string (e.g., "$1.4M", "$250.0K")
 */
export function formatCurrency(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return '—';
    }
    const x = Number(value);
    if (Math.abs(x) >= 1_000_000) {
        return `$${(x / 1_000_000).toFixed(DEFAULT_CONFIG.currencyDecimals)}M`;
    }
    if (Math.abs(x) >= 1_000) {
        return `$${(x / 1_000).toFixed(DEFAULT_CONFIG.currencyDecimals)}K`;
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0,
    }).format(x);
}

/**
 * Format percentage with 1 decimal
 * @param {number} value - Decimal value (0-1)
 * @returns {string} Formatted percentage string
 */
export function formatPercent(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return '—';
    }
    return `${(value * 100).toFixed(1)}%`;
}

// =============================================================================
// Core Metrics Computation
// =============================================================================

/**
 * Compute all dashboard metrics from the population array
 * 
 * @param {Array} population - Array of case records
 * @param {Object} config - Configuration overrides
 * @param {string} scope - 'run' (entire batch) or 'view' (filtered)
 * @returns {Object} Structured metrics object
 */
export function computeMetrics(population = [], config = {}, scope = 'run') {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    // Initialize counters
    let needsClarificationCount = 0;
    let missingRequiredDataCount = 0;
    let cdiRequiredCount = 0; // Phase 9.5
    let approvedCount = 0;
    let deniedCount = 0;
    let safetyContraindicationCount = 0;
    let notEligibleCount = 0;

    let totalValue = 0;
    let approvedValue = 0;     // Revenue Secured
    let deniedValue = 0;       // Cost Avoidance
    let manualValue = 0;       // Pending Adjudication (general)
    let revenueAtRiskValue = 0;// Potential Revenue at Risk (CDI)

    // Single pass through population
    for (const record of population) {
        if (!record) continue;

        const status = safeUpper(record.status);
        const value = Number(record.value) || 0;
        const reason = record.reason || '';

        const bucket = getReviewBucket(status);

        totalValue += value;

        // Count by status category using the Single Source of Truth (buckets)
        switch (bucket) {
            case REVIEW_BUCKET.APPROVED:
                approvedCount++;
                approvedValue += value;
                break;

            case REVIEW_BUCKET.PENDING_CDI:
                cdiRequiredCount++;
                revenueAtRiskValue += value;
                break;

            case REVIEW_BUCKET.DENIED:
                deniedCount++;
                deniedValue += value;
                // Sub-categorize denials (still need granularity for specific KPI breakdown)
                if (isSafetyContraindication(reason)) {
                    safetyContraindicationCount++;
                } else {
                    notEligibleCount++;
                }
                break;

            case REVIEW_BUCKET.MISSING_INFO:
                manualValue += value;
                missingRequiredDataCount++;
                break;

            case REVIEW_BUCKET.NEEDS_REVIEW:
            default:
                manualValue += value;
                // Differentiate needs clarification vs other manual
                // Just count it all as 'needsClarification' for the KPI if not filtered
                needsClarificationCount++;
                break;
        }
    }

    // Derived metrics
    const totalScreened = population.filter(r => r != null).length;
    // NOTE: CDI Required is conceptually a "process hold", separate from general review
    const needsReviewTotal = needsClarificationCount + missingRequiredDataCount;
    const autoResolvedCount = approvedCount + deniedCount;
    // CDI is NOT auto-resolved (requires query), but strictly it's not "needs clarification" in the old sense.
    // However, for total workflow completion, it needs action.

    // Auto-resolution rate
    const autoResolutionRate = totalScreened > 0
        ? autoResolvedCount / totalScreened
        : 0;

    // Hours saved calculations
    // CDI cases are NOT counted as hours saved (Human touch required to send query/review response)
    const hoursPerCase = cfg.minutesPerCasePoint / 60;
    const hoursPerCaseMin = cfg.minutesPerCaseMin / 60;
    const hoursPerCaseMax = cfg.minutesPerCaseMax / 60;

    const hoursSavedPointRaw = autoResolvedCount * hoursPerCase;
    const hoursSavedMin = autoResolvedCount * hoursPerCaseMin;
    const hoursSavedMax = autoResolvedCount * hoursPerCaseMax;

    // Build the return object
    return {
        // Scope
        scope,

        // Counts
        totalScreened,
        approvedCount,
        deniedCount,
        safetyContraindicationCount,
        notEligibleCount,
        needsClarificationCount,
        missingRequiredDataCount,
        cdiRequiredCount, // New
        needsReviewTotal,
        autoResolvedCount,

        // Values (dollars)
        totalValue,
        approvedValue,      // "Revenue Secured"
        deniedValue,        // "Cost Avoidance"
        manualValue,        // "Pending Adjudication"
        revenueAtRiskValue, // "Potential Revenue at Risk"

        // Rates
        autoResolutionRate,

        // Hours saved
        config: {
            minutesPerCasePoint: cfg.minutesPerCasePoint,
            minutesPerCaseMin: cfg.minutesPerCaseMin,
            minutesPerCaseMax: cfg.minutesPerCaseMax,
        },
        hoursSavedPointRaw,
        hoursSavedMin,
        hoursSavedMax,

        // Pre-formatted display values (use these in UI, never do math in components)
        display: {
            totalScreened: String(totalScreened),
            approvedCount: String(approvedCount),
            deniedCount: String(deniedCount),
            needsReviewTotal: String(needsReviewTotal),
            needsClarificationCount: String(needsClarificationCount),
            missingRequiredDataCount: String(missingRequiredDataCount),
            cdiRequiredCount: String(cdiRequiredCount),
            autoResolvedCount: String(autoResolvedCount),
            autoResolutionRate: formatPercent(autoResolutionRate),
            hoursSavedPoint: formatHours(hoursSavedPointRaw),
            hoursSavedRange: `${formatHours(hoursSavedMin)}–${formatHours(hoursSavedMax)}`,
            revenueSecured: formatCurrency(approvedValue),
            costAvoidance: formatCurrency(deniedValue),
            pendingAdjudication: formatCurrency(manualValue),
            revenueAtRisk: formatCurrency(revenueAtRiskValue),
        },

        // Tooltip explanation components
        explain: {
            formula: `Auto-resolved cases × ${cfg.minutesPerCasePoint} min (Staff Time) / 60`,
            formulaExpanded: `${autoResolvedCount} × ${cfg.minutesPerCasePoint} min = ${formatHours(hoursSavedPointRaw)} staff hours`,
            autoResolvedBreakdown: `${approvedCount} approved + ${deniedCount} denied`,
            needsReviewBreakdown: `${needsClarificationCount} needs clarification + ${missingRequiredDataCount} missing data`,
            cdiBreakdown: `${cdiRequiredCount} pending physician query (missing anchor)`,
            sensitivityNote: `Range: ${cfg.minutesPerCaseMin}–${cfg.minutesPerCaseMax} min/case → ${formatHours(hoursSavedMin)}–${formatHours(hoursSavedMax)} hrs`,
            assumptionNote: `Using ${cfg.minutesPerCasePoint} min/case (Governance Assumption: Staff time per complex PA)`,
            riskNote: `Clinically eligible claims stalled by missing administrative codes (E66.x)`,
        },

        // Labels (for KPI cards)
        labels: {
            hoursKpi: 'Staff Hours Saved',
            needsReviewKpi: 'Needs Review',
            autoResolutionKpi: 'Auto-Resolution',
            revenueKpi: 'Revenue Secured',
            riskKpi: 'Revenue at Risk',
        },
    };
}

// =============================================================================
// Legacy Compatibility
// =============================================================================

/**
 * Legacy function for backward compatibility
 * @deprecated Use computeMetrics instead
 */
export function calcAdminHoursSaved(totalOrders, flaggedOrders, minutesPerCase = 25) {
    const autoResolved = totalOrders - flaggedOrders;
    const hours = autoResolved * (minutesPerCase / 60);
    return formatHours(hours);
}
