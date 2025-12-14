import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal, sankeyLeft } from 'd3-sankey';
import { COLORS } from './constants';

const SankeyChart = ({ data }) => {
    const svgRef = useRef(null);
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [tooltip, setTooltip] = useState({ visible: false, x: 0, y: 0, content: null });

    // 2. Handle Responsiveness
    useEffect(() => {
        const resizeObserver = new ResizeObserver(entries => {
            if (!entries || entries.length === 0) return;
            const { width } = entries[0].contentRect;
            setDimensions({ width, height: 400 });
        });

        if (containerRef.current) resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, []);

    const sankeyData = useMemo(() => {
        if (!Array.isArray(data) || data.length === 0) return null;

        const approvedData = [];
        const deniedData = [];
        const manualData = [];

        for (const record of data) {
            if (!record) continue; // Skip null/undefined records
            const status = record?.status;
            if (status === 'APPROVED') {
                approvedData.push(record);
            } else if (status === 'DENIED') {
                deniedData.push(record);
            } else {
                manualData.push(record);
            }
        }

        const safeSum = (records) => records.reduce((acc, curr) => acc + (Number(curr.value) || 0), 0);
        const approvedVal = safeSum(approvedData);
        const deniedVal = safeSum(deniedData);
        const manualVal = safeSum(manualData);

        const rawNodes = [
            { name: "Total Claims Pipeline" },
            { name: "AI Approved" },
            { name: "AI Denied" },
            { name: "Manual Review" },
            { name: "Revenue Recovered" },
            { name: "Cost Avoidance" },
            { name: "Pending Adjudication" }
        ];

        const rawLinks = [
            { source: 0, target: 1, value: approvedVal, count: approvedData.length, type: 'approved' },
            { source: 0, target: 2, value: deniedVal, count: deniedData.length, type: 'denied' },
            { source: 0, target: 3, value: manualVal, count: manualData.length, type: 'manual' },
            { source: 1, target: 4, value: approvedVal, count: approvedData.length, type: 'approved' },
            { source: 2, target: 5, value: deniedVal, count: deniedData.length, type: 'denied' },
            { source: 3, target: 6, value: manualVal, count: manualData.length, type: 'manual' }
        ];

        const filteredLinks = rawLinks.filter((link) => link.value > 0);
        if (filteredLinks.length === 0) return null;

        return { nodes: rawNodes, links: filteredLinks };
    }, [data]);

    // 3. D3 Rendering Logic
    useEffect(() => {
        if (!sankeyData || dimensions.width === 0) return;

        // --- Layout Configuration ---
        const { width, height } = dimensions;
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove();

        const sankeyGenerator = sankey()
            .nodeWidth(4)
            .nodePadding(30)
            .nodeAlign(sankeyLeft)
            .extent([[1, 1], [width - 1, height - 20]]);

        const graph = sankeyGenerator({
            nodes: sankeyData.nodes.map((node) => ({ ...node })),
            links: sankeyData.links.map((link) => ({ ...link })),
        });

        // --- Draw Links ---
        const links = svg.append("g")
            .attr("fill", "none")
            .attr("stroke-opacity", 0.15)
            .style("mix-blend-mode", "multiply")
            .selectAll("path")
            .data(graph.links)
            .join("path")
            .attr("d", sankeyLinkHorizontal())
            .attr("stroke", d => {
                if (d.type === 'denied') return COLORS.denied;
                if (d.type === 'approved') return COLORS.approved;
                if (d.type === 'manual') return COLORS.manual;
                return COLORS.linkBase;
            })
            .attr("stroke-width", d => Math.max(1, d.width))
            .style("transition", "stroke-opacity 0.3s ease");

        // Link Interactions
        links.on("mouseover", function (event, d) {
            d3.select(this).attr("stroke-opacity", 0.6);

            setTooltip({
                visible: true,
                x: event.pageX,
                y: event.pageY,
                content: (
                    <>
                        <div style={{ fontWeight: 600, marginBottom: 4, color: '#94A3B8', fontSize: 11, textTransform: 'uppercase' }}>Flow Logic</div>
                        <div style={{ fontSize: 14, marginBottom: 8 }}>
                            {d.source.name} <span style={{ color: '#94A3B8' }}>→</span> {d.target.name}
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'auto auto', gap: '8px 16px', fontSize: 13 }}>
                            <span style={{ color: '#64748B' }}>Volume:</span>
                            <span style={{ fontFamily: 'JetBrains Mono', textAlign: 'right' }}>{d.count} Claims</span>
                            <span style={{ color: '#64748B' }}>Value:</span>
                            <span style={{ fontFamily: 'JetBrains Mono', textAlign: 'right', fontWeight: 600 }}>
                                ${d.value.toLocaleString()}
                            </span>
                        </div>
                    </>
                )
            });
        })
            .on("mouseout", function () {
                d3.select(this).attr("stroke-opacity", 0.15);
                setTooltip(prev => ({ ...prev, visible: false }));
            });

        // --- Draw Nodes ---
        svg.append("g")
            .selectAll("rect")
            .data(graph.nodes)
            .join("rect")
            .attr("x", d => d.x0)
            .attr("y", d => d.y0)
            .attr("height", d => Math.max(d.y1 - d.y0, 1)) // Ensure min height
            .attr("width", d => d.x1 - d.x0)
            .attr("fill", COLORS.nodes)
            .attr("rx", 2);

        // --- Draw Labels ---
        svg.append("g")
            .attr("font-family", "Inter, sans-serif")
            .attr("font-size", 12)
            .selectAll("text")
            .data(graph.nodes)
            .join("text")
            .attr("x", d => d.x0 < width / 2 ? d.x1 + 10 : d.x0 - 10)
            .attr("y", d => (d.y1 + d.y0) / 2)
            .attr("dy", "0.35em")
            .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
            .style("font-weight", "500")
            .style("fill", "#0F172A")
            .text(d => d.name)
            .append("tspan")
            .attr("fill-opacity", 0.5)
            .attr("font-weight", "400")
            .attr("font-size", 11)
            .attr("x", d => d.x0 < width / 2 ? d.x1 + 10 : d.x0 - 10)
            .attr("dy", "1.4em")
            .text(d => `$${(d.value / 1000000).toFixed(1)}M`);

    }, [sankeyData, dimensions]);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
            <svg ref={svgRef} width={dimensions.width} height={dimensions.height} style={{ overflow: 'visible' }} />

            {/* Tooltip */}
            {tooltip.visible && (
                <div style={{
                    position: 'fixed',
                    left: tooltip.x + 15,
                    top: tooltip.y - 15,
                    background: 'rgba(255, 255, 255, 0.98)',
                    backdropFilter: 'blur(4px)',
                    padding: '12px 16px',
                    border: '1px solid #E2E8F0',
                    borderRadius: '6px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    pointerEvents: 'none',
                    zIndex: 100,
                    minWidth: '200px',
                    color: '#0F172A'
                }}>
                    {tooltip.content}
                </div>
            )}
        </div>
    );
};

import PropTypes from 'prop-types';

SankeyChart.propTypes = {
    data: PropTypes.arrayOf(PropTypes.shape({
        patient_id: PropTypes.string,
        status: PropTypes.string,
        value: PropTypes.number,
        reason: PropTypes.string,
    })),
};

SankeyChart.defaultProps = {
    data: [],
};

export default SankeyChart;
