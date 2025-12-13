"""Simple Flask app for testing"""
from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>News Clustering - Test</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            text-align: center;
            padding: 40px;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 20px;
            border: 1px solid #334155;
            max-width: 600px;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #6366f1, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        p { color: #94a3b8; margin: 15px 0; font-size: 1.1rem; }
        .status {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid #10b981;
            padding: 15px 30px;
            border-radius: 10px;
            color: #10b981;
            font-weight: bold;
            margin-top: 20px;
        }
        a { color: #6366f1; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 News Clustering ML</h1>
        <p>Your Flask application is running successfully!</p>
        <div class="status">✓ Server Status: ONLINE</div>
        <p style="margin-top: 30px;">
            <a href="/api/status">Check API Status →</a>
        </p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/status')
def status():
    return {'status': 'ok', 'message': 'Server is running!'}

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask Server Starting...")
    print("="*50)
    print("\n📱 Open in browser: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
