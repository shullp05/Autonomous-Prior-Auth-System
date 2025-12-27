import os


def enforce_offline():
    """
    Monkeys patches socket and requests to prevent network access.
    Raises RuntimeError if network access is attempted.
    """

    # Check enviroment variable
    if str(os.environ.get("PA_OFFLINE_MODE", "")).lower() != "true":
        return

    print("🚫 OFFLINE MODE ENFORCED: Network access is disabled.")

    def fail_on_network_use(*args, **kwargs):
        raise RuntimeError("Network access forbidden in OFFLINE mode (PA_OFFLINE_MODE=true)")

    # Patch socket
    import socket

    # We must patch socket.socket with a CLASS, not a function, to avoid breaking 'import ssl'
    class DeadSocket:
        def __init__(self, *args, **kwargs):
            fail_on_network_use()
        def connect(self, *args, **kwargs):
            fail_on_network_use()
        def bind(self, *args, **kwargs):
            fail_on_network_use()
        def listen(self, *args, **kwargs):
            fail_on_network_use()
        def accept(self, *args, **kwargs):
            fail_on_network_use()
        def send(self, *args, **kwargs):
            fail_on_network_use()
        def recv(self, *args, **kwargs):
            fail_on_network_use()

    socket.socket = DeadSocket
    socket.create_connection = fail_on_network_use

    # Patch requests if installed
    try:
        import requests
        requests.request = fail_on_network_use
        requests.get = fail_on_network_use
        requests.post = fail_on_network_use
        requests.Session = DeadSocket # Use the dummy class for Session too if mostly used as object
    except ImportError:
        pass

    # Patch urllib just in case
    try:
        import urllib.request
        urllib.request.urlopen = fail_on_network_use
    except ImportError:
        pass
