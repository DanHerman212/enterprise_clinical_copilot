"""Empty a Langfuse MinIO bucket.

Langfuse writes every event and media payload to object storage and keeps only a
reference in ClickHouse. Deleting the database rows therefore leaves the blobs
behind: invisible in the UI, still billed, and still containing patient text.
This walks the two prefixes Langfuse uses and deletes the objects in batches.

Run it on the Langfuse host, on the same docker network as MinIO, with the
credentials in AK/SK. It reports what it deleted and then re-lists to prove the
prefixes are empty.
"""

import os

import boto3

BUCKET = os.environ.get("BUCKET", "langfuse")
PREFIXES = ("events/", "media/")

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["ENDPOINT"],
    aws_access_key_id=os.environ["AK"],
    aws_secret_access_key=os.environ["SK"],
    region_name="us-east-1",
)


def count(prefix: str) -> int:
    total = 0
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        total += len(page.get("Contents", []))
    return total


for prefix in PREFIXES:
    before = count(prefix)
    deleted = 0
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if not keys:
            continue
        s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys})
        deleted += len(keys)
    print(f"{prefix} before={before} deleted={deleted} after={count(prefix)}")

print("done")
