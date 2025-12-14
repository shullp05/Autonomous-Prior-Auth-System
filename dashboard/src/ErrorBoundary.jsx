// ErrorBoundary.jsx - Catches unhandled errors in child components
import React, { Component } from 'react';

class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('ErrorBoundary caught an error:', error, errorInfo);
        this.setState({ errorInfo });
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    padding: '40px',
                    textAlign: 'center',
                    fontFamily: 'Inter, sans-serif',
                    color: '#DC2626'
                }}>
                    <h1>Something went wrong.</h1>
                    <p style={{ color: '#64748B' }}>
                        Please refresh the page or contact support if the issue persists.
                    </p>
                    <details style={{
                        marginTop: '20px',
                        textAlign: 'left',
                        background: '#F8FAFC',
                        padding: '16px',
                        borderRadius: '8px',
                        fontSize: '12px',
                        fontFamily: 'JetBrains Mono, monospace'
                    }}>
                        <summary>Error Details</summary>
                        <pre>{this.state.error?.toString()}</pre>
                        <pre>{this.state.errorInfo?.componentStack}</pre>
                    </details>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
