"""
Simple API server for dashboard manual edits.
Saves letter edit audit trails to output/manual_changes/
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Vite dev server

# Ensure output directory exists
MANUAL_CHANGES_DIR = Path("output/manual_changes")
MANUAL_CHANGES_DIR.mkdir(parents=True, exist_ok=True)

@app.route('/api/save-edit', methods=['POST'])
def save_edit():
    """Save a manual letter edit with audit trail."""
    try:
        data = request.get_json()
        
        if not data or 'patient_id' not in data:
            return jsonify({"error": "Missing patient_id"}), 400
        
        patient_id = data['patient_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create audit record
        audit_record = {
            "patient_id": patient_id,
            "timestamp": data.get('timestamp', datetime.now().isoformat()),
            "original_letter": data.get('original_letter', ''),
            "modified_letter": data.get('modified_letter', ''),
            "status": data.get('status', ''),
            "editor": data.get('editor', 'Manual Edit'),
            "saved_at": datetime.now().isoformat()
        }
        
        # Save to file
        filename = f"edit_{patient_id[:8]}_{timestamp}.json"
        filepath = MANUAL_CHANGES_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(audit_record, f, indent=2)
        
        print(f"[Audit] Saved manual edit: {filepath}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "path": str(filepath)
        })
        
    except Exception as e:
        print(f"[Error] Failed to save edit: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "manual-edits-api"})

if __name__ == '__main__':
    # Security: Default to safe production settings
    # Override with FLASK_DEBUG=true, FLASK_HOST=0.0.0.0 for development
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5052'))
    
    print(f"Starting Manual Edits API server on http://{host}:{port}")
    if debug_mode:
        print("[WARNING] Running in DEBUG mode - do not use in production!")
    app.run(host=host, port=port, debug=debug_mode)
