/**
 * statusConfig.js - Single source of truth for status taxonomy
 * 
 * This module defines all status values, display labels, and grouping logic
 * to prevent taxonomy drift between KPIs, Sankey, and table displays.
 */

// =============================================================================
// Raw Status Values (as they come from backend/data)
// =============================================================================

export const RAW_STATUS = {
    APPROVED: 'APPROVED',
    DENIED: 'DENIED',
    FLAGGED: 'FLAGGED',
    MANUAL_REVIEW: 'MANUAL_REVIEW',
    PROVIDER_ACTION_REQUIRED: 'PROVIDER_ACTION_REQUIRED',
    // Extended denial subtypes (if present in data)
    DENIED_SAFETY: 'DENIED_SAFETY',
    DENIED_CLINICAL: 'DENIED_CLINICAL',
    DENIED_MISSING_INFO: 'DENIED_MISSING_INFO',
    // New Safety Signal (Phase 9)
    SAFETY_SIGNAL_NEEDS_REVIEW: 'SAFETY_SIGNAL_NEEDS_REVIEW',
    // New Coding Integrity (Phase 9.5)
    CDI_REQUIRED: 'CDI_REQUIRED',
};

// =============================================================================
// Display Labels (what users see in pills/tooltips)
// =============================================================================

export const STATUS_DISPLAY = {
    MEETS_CRITERIA: 'Meets Criteria',
    NOT_ELIGIBLE: 'Not Eligible',
    SAFETY_CONTRAINDICATION: 'Safety Contraindication',
    SAFETY_SIGNAL: 'Safety Signal',
    NEEDS_CLARIFICATION: 'Needs Clarification',
    MISSING_REQUIRED_DATA: 'Missing Required Data',
    CDI_REQUIRED: 'CDI Required',
};

// =============================================================================
// Review Buckets (for metric grouping)
// =============================================================================

// =============================================================================
// Review Buckets (for metric grouping)
// =============================================================================

export const REVIEW_BUCKET = {
    APPROVED: 'APPROVED',
    DENIED: 'DENIED',
    PENDING_CDI: 'PENDING_CDI',
    NEEDS_REVIEW: 'NEEDS_REVIEW',
    MISSING_INFO: 'MISSING_INFO',
    // Fallback/Legacy (should be avoided in new logic, but kept for safety if needed)
    AUTO_RESOLVED: 'AUTO_RESOLVED',
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Safely uppercase a string value
 */
export function safeUpper(s) {
    return String(s ?? '').toUpperCase();
}

/**
 * Detect if a denial is due to safety contraindication (HARD STOP)
 */
export function isSafetyContraindication(reason) {
    return reason && String(reason).toUpperCase().includes('HARD STOP');
}

/**
 * Get the display label for a status
 * @param {string} status - Raw status from data
 * @param {string} reason - Optional reason field for HARD STOP detection
 * @returns {string} Human-readable display label
 */
export function getStatusDisplayLabel(status, reason = '') {
    const s = safeUpper(status);

    if (s === RAW_STATUS.APPROVED) {
        return STATUS_DISPLAY.MEETS_CRITERIA;
    }

    if (s === RAW_STATUS.CDI_REQUIRED) {
        return STATUS_DISPLAY.CDI_REQUIRED;
    }

    if (s === RAW_STATUS.SAFETY_SIGNAL_NEEDS_REVIEW) {
        return STATUS_DISPLAY.SAFETY_SIGNAL;
    }

    if (s === RAW_STATUS.FLAGGED || s === RAW_STATUS.MANUAL_REVIEW) {
        return STATUS_DISPLAY.NEEDS_CLARIFICATION;
    }

    if (s === RAW_STATUS.PROVIDER_ACTION_REQUIRED) {
        return STATUS_DISPLAY.MISSING_REQUIRED_DATA;
    }

    if (
        s === RAW_STATUS.DENIED ||
        s === RAW_STATUS.DENIED_SAFETY ||
        s === RAW_STATUS.DENIED_CLINICAL ||
        s === RAW_STATUS.DENIED_MISSING_INFO
    ) {
        return isSafetyContraindication(reason)
            ? STATUS_DISPLAY.SAFETY_CONTRAINDICATION
            : STATUS_DISPLAY.NOT_ELIGIBLE;
    }

    // Fallback: return original status
    return status;
}

/**
 * Get the CSS class for a status pill
 * @param {string} status - Raw status from data
 * @param {string} reason - Optional reason field for HARD STOP detection
 * @returns {string} CSS class name
 */
export function getStatusPillClass(status, reason = '') {
    const s = safeUpper(status);

    if (s === RAW_STATUS.APPROVED) {
        return 'status-approved';
    }

    if (s === RAW_STATUS.CDI_REQUIRED) {
        return 'status-cdi_required';
    }

    if (s === RAW_STATUS.SAFETY_SIGNAL_NEEDS_REVIEW) {
        return 'status-safety_signal';
    }

    if (s === RAW_STATUS.FLAGGED || s === RAW_STATUS.MANUAL_REVIEW) {
        return 'status-flagged';
    }

    if (s === RAW_STATUS.PROVIDER_ACTION_REQUIRED) {
        return 'status-provider_action_required';
    }

    if (
        s === RAW_STATUS.DENIED ||
        s === RAW_STATUS.DENIED_SAFETY ||
        s === RAW_STATUS.DENIED_CLINICAL ||
        s === RAW_STATUS.DENIED_MISSING_INFO
    ) {
        return isSafetyContraindication(reason)
            ? 'status-safety_contraindication'
            : 'status-not_eligible';
    }

    return `status-${s.toLowerCase()}`;
}

/**
 * Get the review bucket for a case (used for metric grouping)
 * @param {string} status - Raw status from data
 * @returns {string} One of REVIEW_BUCKET values
 */
export function getReviewBucket(status) {
    const s = safeUpper(status);

    if (s === RAW_STATUS.APPROVED) {
        return REVIEW_BUCKET.APPROVED;
    }

    if (s === RAW_STATUS.CDI_REQUIRED) {
        return REVIEW_BUCKET.PENDING_CDI;
    }

    if (
        s === RAW_STATUS.DENIED ||
        s === RAW_STATUS.DENIED_SAFETY ||
        s === RAW_STATUS.DENIED_CLINICAL ||
        s === RAW_STATUS.DENIED_MISSING_INFO
    ) {
        return REVIEW_BUCKET.DENIED;
    }

    if (s === RAW_STATUS.PROVIDER_ACTION_REQUIRED) {
        return REVIEW_BUCKET.MISSING_INFO;
    }

    // Default catch-all for ambiguity or manual checks
    // Includes: FLAGGED, MANUAL_REVIEW, SAFETY_SIGNAL_NEEDS_REVIEW
    return REVIEW_BUCKET.NEEDS_REVIEW;
}

/**
 * Check if a case needs clarification (ambiguous terms)
 */
export function isNeedsClarification(status) {
    const s = safeUpper(status);
    return s === RAW_STATUS.FLAGGED || s === RAW_STATUS.MANUAL_REVIEW;
}

/**
 * Check if a case is missing required data
 */
export function isMissingRequiredData(status) {
    const s = safeUpper(status);
    return s === RAW_STATUS.PROVIDER_ACTION_REQUIRED;
}

/**
 * Determines the Sankey category for a case
 * @param {string} status - Raw status from data
 * @returns {'approved' | 'denied' | 'manual' | 'cdi'} Sankey category
 */
export function getSankeyCategory(status) {
    const bucket = getReviewBucket(status);

    if (bucket === REVIEW_BUCKET.APPROVED) {
        return 'approved';
    }

    if (bucket === REVIEW_BUCKET.PENDING_CDI) {
        return 'cdi'; // Note: The user might want this separated or manual. Existing SankeyChart handles 'cdi' implicitly in 'manual' or explicit logic?
        // Checking SankeyChart.jsx: it looked for 'approved', 'denied', else 'manual'. 
        // But logic inside it counts CDI. 
        // Let's keep returning 'cdi' if we want to support it, or map to 'manual' if that's what the chart expects.
        // Re-reading SankeyChart.jsx: "if (category === 'approved') ... else if ('denied') ... else { manualData.push }"
        // So anything not 'approved' or 'denied' goes to manual.
        // But wait, if I return 'cdi', does it go to manual? Yes.
        // So returning 'cdi' is fine, it falls through to manual node.
    }

    if (bucket === REVIEW_BUCKET.DENIED) {
        return 'denied';
    }

    return 'manual';
}
