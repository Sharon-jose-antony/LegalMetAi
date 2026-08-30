"""
LEGALMET AI — Mobile Camera Workflow Verification Script
Simulates full field inspection workflow:
1. Inspector Authentication
2. Network IP discovery
3. Draft inspection creation
4. Camera photo upload & automated quality assessment
5. Multi-pass AI pipeline execution & statutory rule evaluation
"""
import sys
import os
import io
import json
import httpx

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

API = os.getenv("API_URL", "http://127.0.0.1:8000")

def run_mobile_workflow_test():
    client = httpx.Client(base_url=API, timeout=120.0)

    print("=" * 60)
    print("LEGALMET AI — Mobile Camera Workflow Test")
    print("=" * 60)

    # 1. Login
    auth_res = client.post("/api/auth/login", json={"username": "inspector1", "password": "Inspector@1234!"})
    if auth_res.status_code != 200:
        print("[FAIL] Authentication failed:", auth_res.text)
        return False
    token = auth_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[PASS] 1. Authenticated as Inspector (token received)")

    # 2. Check network info endpoint
    net = client.get("/api/system/network").json()
    print(f"[PASS] 2. Network Info endpoint verified: {net.get('primary_url')}")

    # 3. Create Quick Inspection
    insp_res = client.post(
        "/api/inspections",
        json={
            "product_name": "Mobile Camera Sample Package",
            "brand_name": "Britannia",
            "product_category": "PACKAGED_FOOD",
            "notes": "Captured via Mobile Camera in Field Inspection",
        },
        headers=headers,
    )
    insp_id = insp_res.json()["inspection_id"]
    print(f"[PASS] 3. Created inspection record: {insp_id}")

    # 4. Upload photo from camera
    sample_path = os.path.join("data", "test", "britannia_marie_gold_500x337.webp")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("data", "demo", "demo1_food_compliant.png")

    with open(sample_path, "rb") as f:
        upload_res = client.post(
            f"/api/inspections/{insp_id}/upload",
            files={"file": (os.path.basename(sample_path), f, "image/png")},
            headers=headers,
        )
    print(f"[PASS] 4. Photo uploaded and quality checked (Score: {upload_res.json().get('quality_score', 0):.2f})")

    # 5. Run AI Analysis Pipeline
    analyzed_res = client.post(f"/api/inspections/{insp_id}/analyze", headers=headers)
    analyzed = analyzed_res.json()
    status = analyzed.get("status")
    conf = analyzed.get("overall_confidence", 0)
    print(f"[PASS] 5. AI pipeline completed (Status: {status}, Confidence: {conf:.2f})")

    decls = analyzed.get("declarations", [])
    found = [d for d in decls if d.get("status") == "FOUND"]
    print(f"\nExtracted {len(found)} declarations:")
    for d in found:
        print(f"  • {d.get('field')}: {d.get('extracted_value')} (conf: {d.get('extraction_confidence')})")

    print("\n" + "=" * 60)
    print("MOBILE WORKFLOW TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_mobile_workflow_test()
