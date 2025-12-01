import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal } from 'd3-sankey';

const SankeyChart = ({ data }) => {
    const svgRef = useRef(null);

    useEffect(() => {
        if (!data || data.length === 0) return;

        // 1. Transform Data for Sankey
        // Nodes: Total -> [Approved, Denied] -> [Revenue, Savings]
        const totalValue = data.reduce((acc, curr) => acc + curr.value, 0);
        const approved = data.filter(d => d.status === 'APPROVED');
        const denied = data.filter(d => d.status === 'DENIED');
        
        const approvedVal = approved.reduce((acc, curr) => acc + curr.value, 0);
        const deniedVal = denied.reduce((acc, curr) => acc + curr.value, 0);

        const nodes = [
            { name: "Total Claims Pipeline" }, // 0
            { name: "AI Approved" },           // 1
            { name: "AI Denied" },             // 2
            { name: "Revenue Recovered" },     // 3
            { name: "Cost Avoidance" }         // 4
        ];

        const links = [
            { source: 0, target: 1, value: approvedVal },
            { source: 0, target: 2, value: deniedVal },
            { source: 1, target: 3, value: approvedVal },
            { source: 2, target: 4, value: deniedVal }
        ];

        // 2. Setup D3 SVG
        const width = 800;
        const height = 400;
        const svg = d3.select(svgRef.current);
        svg.selectAll("*").remove(); // Clear previous

        const { nodes: sankeyNodes, links: sankeyLinks } = sankey()
            .nodeWidth(15)
            .nodePadding(20)
            .extent([[1, 1], [width - 1, height - 6]])({
                nodes: nodes.map(d => Object.assign({}, d)),
                links: links.map(d => Object.assign({}, d))
            });

        // 3. Draw Links
        svg.append("g")
            .attr("fill", "none")
            .attr("stroke-opacity", 0.5)
            .selectAll("path")
            .data(sankeyLinks)
            .join("path")
            .attr("d", sankeyLinkHorizontal())
            .attr("stroke", d => d.target.name.includes("Denied") ? "#ff6b6b" : "#4ecdc4")
            .attr("stroke-width", d => Math.max(1, d.width))
            .append("title")
            .text(d => `${d.source.name} → ${d.target.name}\n$${d.value.toLocaleString()}`);

        // 4. Draw Nodes
        svg.append("g")
            .selectAll("rect")
            .data(sankeyNodes)
            .join("rect")
            .attr("x", d => d.x0)
            .attr("y", d => d.y0)
            .attr("height", d => d.y1 - d.y0)
            .attr("width", d => d.x1 - d.x0)
            .attr("fill", "#333")
            .append("title")
            .text(d => `${d.name}\n$${d.value.toLocaleString()}`);

        // 5. Add Labels
        svg.append("g")
            .attr("font-family", "sans-serif")
            .attr("font-size", 12)
            .selectAll("text")
            .data(sankeyNodes)
            .join("text")
            .attr("x", d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
            .attr("y", d => (d.y1 + d.y0) / 2)
            .attr("dy", "0.35em")
            .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
            .text(d => d.name)
            .attr("font-weight", "bold");

    }, [data]);

    return (
        <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '20px', background: '#fff' }}>
            <h3 style={{marginTop: 0, color: '#444'}}>Revenue Flow Analysis</h3>
            <svg ref={svgRef} width={800} height={400} />
        </div>
    );
};

export default SankeyChart;
