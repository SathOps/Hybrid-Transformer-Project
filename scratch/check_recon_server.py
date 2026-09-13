import urllib.request
import re
import ssl
import sys

url = "https://2023-ciciot.cicresearch.ca/csv/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Fetching {url}...")
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, context=ctx).read().decode("utf-8")
    links = re.findall(r'href="([^"]+)"', html)
    print(f"Total links found on page: {len(links)}")
    recon_links = [l for l in links if "recon" in l.lower()]
    print(f"Recon links found: {len(recon_links)}")
    for r in recon_links:
        print("  -", r)
    
    print("\nAll directory links on server:")
    for l in sorted(set(links)):
        if l.endswith("/") and not l.startswith("?"):
            print("  -", l)
except Exception as e:
    print("Error:", e)
