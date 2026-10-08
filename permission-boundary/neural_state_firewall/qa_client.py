"""Ask the local document-Q&A pilot; print its response and retrieval provenance."""
import argparse
import http.client
import json
import sys

from .artifacts import loads
from .pilot import load_token


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=70)
    args = parser.parse_args(argv)
    connection = None
    try:
        token = load_token(args.token_file)
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=args.timeout)
        connection.request("POST", "/v1/respond", json.dumps({"task": args.task}),
                           {"Content-Type": "application/json", "Authorization": "Bearer " + token})
        response = connection.getresponse()
        result = loads(response.read(1048577).decode("utf-8"))
        print(json.dumps({"http_status": response.status, "response": result}, indent=2, allow_nan=False))
        return 0 if response.status == 200 else 2
    except (OSError, ValueError, http.client.HTTPException):
        print("Could not read a valid response from the local pilot; check the service and token file.", file=sys.stderr)
        return 2
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
