"""rag_endpoint_status.py — show the RAG index endpoint + deployed indexes."""

from google.cloud.aiplatform_v1 import IndexEndpointServiceClient

LOCATION = "us-east1"
PARENT = f"projects/trim-icon-498815-a0/locations/{LOCATION}"


def main() -> None:
    c = IndexEndpointServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"})
    found = False
    for ep in c.list_index_endpoints(parent=PARENT):
        found = True
        print("endpoint:", ep.display_name)
        for d in ep.deployed_indexes:
            print(f"   deployed id={d.id} index={d.index.split('/')[-1]}")
    if not found:
        # "Nothing is deployed" is the answer this script exists to give, and it
        # is the answer a teardown produces. Printing nothing for it left the
        # reader unable to tell that from a script that failed silently.
        print(f"no index endpoints in {LOCATION} — nothing is billing hourly")
        print("(the index itself is storage and is not listed here: "
              "gcloud ai indexes list --region us-east1)")


if __name__ == "__main__":
    main()
