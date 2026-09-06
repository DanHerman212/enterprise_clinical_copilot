"""rag_endpoint_status.py — show the RAG index endpoint + deployed indexes."""

from google.cloud.aiplatform_v1 import IndexEndpointServiceClient

LOCATION = "us-east1"
PARENT = f"projects/trim-icon-498815-a0/locations/{LOCATION}"


def main() -> None:
    c = IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})
    for ep in c.list_index_endpoints(parent=PARENT):
        print("endpoint:", ep.display_name)
        for d in ep.deployed_indexes:
            print(f"   deployed id={d.id} index={d.index.split('/')[-1]}")


if __name__ == "__main__":
    main()
