######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: net_setup.py
# function: make the huggingface / torch hub downloads work on this machine.
#
# two problems are fixed here:
#   1. the openssl of the conda env must find the CA bundle of the env; the
#      check is kept as a safety net because the env was copied from another
#      machine once and its openssl pointed at a path that does not exist
#      anymore.
#   2. 'huggingface.co' refuses the connections on this host, the mirror
#      'hf-mirror.com' is used instead.
#
# the module must be imported *before* huggingface_hub / torch are imported.
######################################################
import os
import sys

# the mirror that is reachable from this machine
HF_MIRROR = "https://hf-mirror.com"


def _conda_ssl_dirs():
    """Return the ``ssl`` folders of the active conda env."""
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    return [
        os.path.join(prefix, "ssl"),
        os.path.join(prefix, "ssl", "certs"),
    ]


def setup_ssl_certificates():
    """Point openssl at the CA bundle of the current conda env.

    The openssl of the env finds its certificates on its own; the function
    only sets the environment variables when the default paths are broken
    (missing or pointing at a path of another machine).

    Returns the path of the certificate file that is used, or None when
    nothing had to be changed.
    """
    ssl_dirs = _conda_ssl_dirs()

    # the cert file of the current env
    cert_file = None
    for ssl_dir in ssl_dirs:
        for name in ("cert.pem", "cacert.pem"):
            candidate = os.path.join(ssl_dir, name)
            if os.path.exists(candidate):
                cert_file = candidate
                break
        if cert_file:
            break

    if cert_file is None:
        return None

    # only override when the current value is broken (missing or pointing at
    # a path of another machine)
    current = os.environ.get("SSL_CERT_FILE", "")
    if not current or not os.path.exists(current):
        os.environ["SSL_CERT_FILE"] = cert_file

    cert_dir = ssl_dirs[-1]
    current_dir = os.environ.get("SSL_CERT_DIR", "")
    if not current_dir or not os.path.isdir(current_dir):
        os.environ["SSL_CERT_DIR"] = cert_dir

    return os.environ.get("SSL_CERT_FILE")


def setup_hf_endpoint(use_mirror=True):
    """Use the reachable mirror instead of the unreachable huggingface.co.

    Returns the endpoint that is used, or None when nothing was changed.
    """
    if not use_mirror:
        return None

    current = os.environ.get("HF_ENDPOINT", "")
    if current:
        # the user already configured an endpoint, keep it
        return current

    os.environ["HF_ENDPOINT"] = HF_MIRROR
    return HF_MIRROR


def setup_wandb_observability(enable=False):
    """Disable the wandb observability (the system metrics).

    The system metrics (gpu / cpu) are sent by an opentelemetry exporter that
    runs in the wandb service; when the wandb api is not reachable the export
    fails with a '403 Forbidden' every minute.  The exporter is disabled with
    the 'WANDB_ERROR_REPORTING' env var, which must be set *before* the wandb
    service is started.

    Returns the value that is used, or None when nothing was changed.
    """
    current = os.environ.get("WANDB_ERROR_REPORTING", "")
    if current:
        # the user already configured it, keep it
        return current

    value = "True" if enable else "False"
    os.environ["WANDB_ERROR_REPORTING"] = value
    return value


def setup_wandb_timeout(timeout=60):
    """Raise the timeout of the wandb http requests.

    The wandb service answers an api request with a hardcoded timeout of 10
    seconds; the mirror that is used from china is often slower than that, so
    the request fails with 'the service process is busy and did not respond in
    time'.  The timeout is raised with the 'WANDB_HTTP_TIMEOUT' env var, which
    must be set *before* the wandb service is started.

    Returns the value that is used, or None when nothing was changed.
    """
    current = os.environ.get("WANDB_HTTP_TIMEOUT", "")
    if current:
        # the user already configured it, keep it
        return current

    os.environ["WANDB_HTTP_TIMEOUT"] = str(int(timeout))
    return str(int(timeout))


def setup(use_mirror=True, wandb_observability=False, wandb_timeout=60):
    """Apply both fixes; call this before importing torch / huggingface_hub."""
    cert_file = setup_ssl_certificates()
    endpoint = setup_hf_endpoint(use_mirror)
    observability = setup_wandb_observability(wandb_observability)
    http_timeout = setup_wandb_timeout(wandb_timeout)
    return {
        "ssl_cert_file": cert_file,
        "hf_endpoint": endpoint,
        "wandb_observability": observability,
        "wandb_http_timeout": http_timeout,
    }


if __name__ == "__main__":
    info = setup()
    print(f"SSL_CERT_FILE         = {info['ssl_cert_file']}")
    print(f"HF_ENDPOINT           = {info['hf_endpoint']}")
    print(f"WANDB_ERROR_REPORTING = {info['wandb_observability']}")
    print(f"WANDB_HTTP_TIMEOUT    = {info['wandb_http_timeout']}")
