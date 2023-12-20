import socket
import logging
import struct

from udao_trace.utils.logging import _get_logger

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    logger.info(f"received message length: {msglen}")

    # Read the message data
    msg_data = recvall(sock, msglen)
    return msg_data.decode('utf-8')


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        logger.debug(f"current data length: {len(data)}, target: {n}")
        packet = sock.recv(n - len(data))
        if not packet:
            logger.debug(f"packet is empty, returning None")
            return None
        data.extend(packet)
    logger.debug(f"received data: {data}, with length: {len(data)}")
    return data


host = "localhost"
port = 12345 # UDAO in 9-grid input

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))
sock.listen(1)

logger = _get_logger(
    name="server",
    std_level=logging.INFO,
    file_level=logging.DEBUG,
    log_file_path="server.log",
)

try:
    while True:
        logger.info(f"Server listening on {host}:{port}")
        conn, addr = sock.accept()
        logger.info(f"Connected by {addr}")

        while True:
            message = recv_msg(conn)
            logger.info(f"Received message: {message}")
            if not message:
                logger.warning(f"No message received, disconnecting {addr}")
                break
            response = "xxx\n"
            conn.sendall(response.encode("utf-8"))
            logger.info(f"Sent response: {response}")

        conn.close()
except Exception as e:
    logger.exception(f"Exception occurred: {e}")
    sock.close()