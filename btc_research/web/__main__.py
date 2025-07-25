"""
Main entry point for Flask web application.
Run with: python -m btc_research.web
"""

import os
import sys
from .app import app, socketio

def main():
    """Main entry point for the Flask application."""
    # Set default configuration
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Paper Trading Dashboard on http://{host}:{port}")
    print(f"Debug mode: {'enabled' if debug else 'disabled'}")
    print(f"API Base URL: {os.environ.get('API_BASE_URL', 'http://localhost:8001')}")
    
    try:
        # Run the Flask app with SocketIO support
        socketio.run(app, 
                    host=host, 
                    port=port,
                    debug=debug,
                    use_reloader=debug,
                    log_output=True,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down Flask application...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Flask application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()