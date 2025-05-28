# publisher_py3.py (Python 3)
import socket
import threading
import json

class Publisher:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.clients = []
        self.lock = threading.Lock()
        
        # Start server thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

    def _run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"[PY3] Publisher listening on {self.host}:{self.port}")
            
            while True:
                conn, addr = s.accept()
                with self.lock:
                    self.clients.append(conn)
                print(f"[PY3] Subscriber connected: {addr}")

    def publish(self, message):
        """Send JSON message to all subscribers."""
        msg_json = json.dumps(message).encode('utf-8')
        with self.lock:
            to_remove = []
            for i, client in enumerate(self.clients):
                try:
                    client.sendall(msg_json)
                except (BrokenPipeError, ConnectionResetError):
                    to_remove.append(i)
                    client.close()
            
            # Remove disconnected clients
            for idx in reversed(to_remove):
                del self.clients[idx]

# Example usage
if __name__ == "__main__":
    publisher = Publisher()
    
    # Simulate motion generation
    while True:
        cmd = input("Enter motion command (or 'exit'): ")
        if cmd.lower() == 'exit':
            break
        publisher.publish({"command": cmd})
        print(f"[PY3] Sent: {cmd}")