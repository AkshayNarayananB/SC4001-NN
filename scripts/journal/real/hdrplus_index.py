"""Index the full HDR+ burst dataset (gs://hdrplusdata/20171106/bursts) by exposure, without downloading it:
reads ExposureTime / ISO / model from the first 256 KB of each burst's payload_N000.dng (HTTP range request).

    python -m scripts.journal.real.hdrplus_index --out /mnt/Data/akshay/real/hdrplus/index.csv
"""
import argparse
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import exifread
import pandas as pd

API = "https://storage.googleapis.com/storage/v1/b/hdrplusdata/o"
GET = "https://storage.googleapis.com/hdrplusdata/"


def list_bursts():
    out, tok = [], None
    while True:
        u = f"{API}?prefix=20171106/bursts/&delimiter=/&fields=prefixes,nextPageToken&maxResults=1000"
        d = json.load(urllib.request.urlopen(u + (f"&pageToken={tok}" if tok else "")))
        out += d.get("prefixes", [])
        tok = d.get("nextPageToken")
        if not tok:
            return out


def meta(prefix):
    try:
        lst = json.load(urllib.request.urlopen(f"{API}?prefix={prefix}&fields=items(name,size)"))["items"]
        n = sum(i["name"].endswith(".dng") for i in lst)
        size = sum(int(i["size"]) for i in lst)
        req = urllib.request.Request(GET + prefix + "payload_N000.dng", headers={"Range": "bytes=0-262143"})
        tags = exifread.process_file(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()), details=False)
        g = lambda k: str(tags.get(k, ""))
        et = tags.get("EXIF ExposureTime") or tags.get("Image ExposureTime")
        iso = tags.get("EXIF ISOSpeedRatings") or tags.get("Image ISOSpeedRatings")
        etv = float(et.values[0].num) / et.values[0].den if et else None
        return dict(burst=prefix.split("/")[-2], n_frames=n, bytes=size, exposure_s=etv,
                    iso=float(str(iso)) if iso else None, model=g("Image Model"))
    except Exception as e:
        return dict(burst=prefix.split("/")[-2], error=repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bursts = list_bursts()
    print(len(bursts), "bursts", flush=True)
    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(meta, bursts))
    df = pd.DataFrame(rows)
    df["exposure_x_iso"] = df.exposure_s * df.iso
    df.to_csv(a.out, index=False)
    print(df.describe(), "\nerrors:", df.get("error", pd.Series()).notna().sum())


if __name__ == "__main__":
    main()
