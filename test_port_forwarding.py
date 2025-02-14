import http.server
import socketserver
import socket

# Define the HTML content as a string
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Port Forwarding Test</title>
</head>
<body>
    <h1>Port forwarding working</h1>
</body>
</html>
"""

# Set up a simple HTTP server
PORT = 19916  # You can change this port to any other available port

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the static HTML content directly without a file
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

# Set up the server using the selected port
http = False

if http:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving on port {PORT}. Visit http://localhost:{PORT} to check port forwarding.")
        httpd.serve_forever()
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("", PORT))
    s.sendall(b'Hello, world')