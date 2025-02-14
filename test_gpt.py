import socket

ip = 'localhost'
port = 19916

# Create a TCP client and connect to the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((ip, port))
    s.sendall(b'Hello, server!')
    data = s.recv(1024)

print(f"Received: {data.decode()}")