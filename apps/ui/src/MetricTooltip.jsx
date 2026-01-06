/**
 * MetricTooltip.jsx - Click-to-pin tooltip component
 * 
 * Features:
 * - Hover to preview (existing behavior)
 * - Click to pin/unpin
 * - Click outside to close
 * - ESC key to close
 * - Keyboard accessible (Tab + Enter/Space)
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import { Info } from 'lucide-react';

export default function MetricTooltip({ children, iconSize = 14 }) {
    const [isPinned, setIsPinned] = useState(false);
    const triggerRef = useRef(null);
    const tooltipRef = useRef(null);

    // Handle click outside to close
    useEffect(() => {
        if (!isPinned) return;

        function handleClickOutside(event) {
            if (
                triggerRef.current &&
                !triggerRef.current.contains(event.target) &&
                tooltipRef.current &&
                !tooltipRef.current.contains(event.target)
            ) {
                setIsPinned(false);
            }
        }

        function handleEscape(event) {
            if (event.key === 'Escape') {
                setIsPinned(false);
            }
        }

        document.addEventListener('mousedown', handleClickOutside);
        document.addEventListener('keydown', handleEscape);

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            document.removeEventListener('keydown', handleEscape);
        };
    }, [isPinned]);

    const handleToggle = useCallback((event) => {
        event.preventDefault();
        event.stopPropagation();
        setIsPinned((prev) => !prev);
    }, []);

    const handleKeyDown = useCallback((event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            setIsPinned((prev) => !prev);
        }
    }, []);

    return (
        <div
            ref={triggerRef}
            className={`metric-tooltip-trigger ${isPinned ? 'is-pinned' : ''}`}
        >
            <button
                type="button"
                className="metric-tooltip-icon"
                onClick={handleToggle}
                onKeyDown={handleKeyDown}
                aria-expanded={isPinned}
                aria-haspopup="true"
                aria-label="Show calculation details"
            >
                <Info size={iconSize} />
            </button>

            <div
                ref={tooltipRef}
                className={`metric-tooltip ${isPinned ? 'is-pinned' : ''}`}
                role="tooltip"
                aria-hidden={!isPinned}
            >
                {/* Close button when pinned */}
                {isPinned && (
                    <button
                        type="button"
                        className="metric-tooltip-close"
                        onClick={() => setIsPinned(false)}
                        aria-label="Close tooltip"
                    >
                        ×
                    </button>
                )}
                {children}
            </div>
        </div>
    );
}

MetricTooltip.propTypes = {
    children: PropTypes.node.isRequired,
    iconSize: PropTypes.number,
};
