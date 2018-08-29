'''A module with utilities for passing messages between processes on the same system.

References:
  - https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data

'''

# IMPORTS ==========================================================================================

# python
import socket, struct, platform, pickle

# FUNCTIONS ========================================================================================

def InitClientSocket():
  '''Initializes a client socket and immediately tries to connect to the server.
  - Returns sock: A client socket ready to send/receive. Be sure to call sock.close() when done with it.
  '''

  port = 9999

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  host = socket.gethostname()
  sock.connect((host, port))

  return sock

def InitServerSocket():
  '''Initializes a server socket and waits for a connection.
  - Returns sock: A server socket ready to send/receive. Be sure to call sock.close() when done with it.
  '''

  # parameters
  port = 9999
  nConnections = 1

  # initialize
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  host = socket.gethostname()
  sock.bind((host, port))
  sock.listen(nConnections)

  # listen for connections
  sock, address = sock.accept()
  print("Received a connection from {}.".format(str(address)))

  return sock

def SendMessage(sock, data):
  '''Sends the entire message over the socket. Receiver must use network.ReceiveMessage().
  - Input sock: The socket to send data over.
  - Input data: A Pyhthon object to serialize and send.
  - Returns None.
  '''

  data = pickle.dumps(data, protocol=2)
  data = struct.pack('>I', len(data)) + data
  sock.sendall(data)

def ReceiveMessage(sock):
  '''Receives the entire message over the socket. Sender must use network.SendMessage().
  - Input sock: The socket to listen for packets over.
  - Returns data: The received Python object.
  '''

  # receive message size
  data = b''
  while len(data) < 4:
    packet = sock.recv(4 - len(data))
    if not packet: return None
    data += packet
  nBytes = struct.unpack('>I', data)[0]

  # receive data
  data = b''
  while len(data) < nBytes:
    packet = sock.recv(nBytes - len(data))
    if not packet: return None
    data += packet

  if int(platform.python_version()[0]) >= 3:
    data = pickle.loads(data, encoding='bytes')
  else:
    data = pickle.loads(data)

  return data